import torch
import numpy as np
import torch.optim as optim
import torch.multiprocessing as mp
import time
# import os
# from collections import namedtuple
# from itertools import count

from torch.distributions import Categorical

from src.env.Wordle import WordleEnv
from src.models.ActorCritic import Actor, Critic
from src.models.Adversary import Adversary
from src.utils.helpers import build_action_mask
def main():
    device = torch.device("cpu")
    torch.set_default_device(device)


    def select_action_actor(actor, state, available_actions, action_size):

        # Create a mask tensor for previously chosen actions
        mask = build_action_mask(action_size, available_actions, device)

        # Add the mask to the actor output and sample from the distribution
        actor_output = actor(state)
        masked_output = actor_output + mask
        masked_distribution = Categorical(logits=masked_output)
        return masked_distribution.sample().view(1, 1)

    def run_episode(actor, critic, optimizer_actor, optimizer_critic, env):
        """
        Debug version: traces exactly where execution may stall.
        """
        total_reward = 0.0
        done = False
        step_idx = 0

        episodic_actor_loss = torch.zeros((), device=device)
        episodic_critic_loss = torch.zeros((), device=device)

        print("[run_episode] start", flush=True)

        while not done:
            step_idx += 1

            state = torch.tensor(env.current_state, device=device, dtype=torch.float32)

            actor_output = actor(state)
            value = critic(state).squeeze()  # scalar-like tensor

            mask = torch.full((env.action_size,), -1e9, device=device)
            for idx in env.available_actions:
                mask[idx] = 0.0

            masked_output = actor_output + mask
            dist = Categorical(logits=masked_output)

            action = dist.sample()  # scalar action tensor


            next_state, reward, done, won, attempts = env.step(action.item())


            next_state_t = torch.tensor(next_state, device=device, dtype=torch.float32)
            next_value = critic(next_state_t).squeeze()

            reward_t = torch.tensor(float(reward), device=device)
            done_t = torch.tensor(float(done), device=device)

            td_error = reward_t + 0.99 * next_value * (1.0 - done_t) - value
            log_prob = dist.log_prob(action).squeeze()



            episodic_actor_loss = (
                episodic_actor_loss - (log_prob * td_error.detach()).squeeze()
            )
            episodic_critic_loss = episodic_critic_loss + td_error.pow(2).squeeze()
            total_reward += float(reward)


        optimizer_actor.zero_grad()
        episodic_actor_loss.backward()
        optimizer_actor.step()

        optimizer_critic.zero_grad()
        episodic_critic_loss.backward()
        optimizer_critic.step()

        return total_reward

    def train(
        num_episodes=1000,
        protagonist_mode="easy",
        antagonist_mode="easy",
        vocab_path="src/data/answers.txt",
        alpha=0.01,
        epsilon=0.2,
        actor_lr=0.01,
        critic_lr=0.01,
    ):
        np.random.seed(439)
        torch.manual_seed(439)

        env = WordleEnv(mode=protagonist_mode)
        actor = Actor(env.state_size, env.action_size).to(device)
        critic = Critic(env.state_size).to(device)

        optimizer_actor = optim.Adam(actor.parameters(), lr=actor_lr)
        optimizer_critic = optim.Adam(critic.parameters(), lr=critic_lr)
        adversary = Adversary(vocab_path=vocab_path, alpha=alpha, epsilon=epsilon)

        completed_episodes = 0
        start = time.time()

        while completed_episodes < int(num_episodes):
            adversary_action = adversary.select_action()
            target = env.vocab[adversary_action]
            env.target_word = target

            total_reward = run_episode(
                actor, critic, optimizer_actor, optimizer_critic, env
            )
            adversary.observe(adversary_action, total_reward)

            completed_episodes += 1

            # Lightweight progress print
            if completed_episodes % 10 == 0:
                print(
                    f"episode={completed_episodes} steps={steps} reward={total_reward:.3f} target={target}",
                    flush=True,
                )

        end = time.time()
        print(f"training_time_sec={end - start:.2f}")

    train()

main()
# def collect_episode_trajectories(
#         seed,
#         target,
#         mode,
#         actor_p,
#         actor_a,
# ):
#     """
#     Runs one episode for the protagonist/antagonist and returns experience data.
#     """
#
#     torch.set_num_threads(1)
#
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#
#     env = WordleEnv(mode=mode)
#     env.target_word = target
#
#     def collect_agent_episode(actor, batch_size=1):
#         episodes = [[] for _ in range(batch_size)]
#         done = False
#         for episode in range(batch_size):
#             while not done:
#                 state = env.current_state.copy()
#                 available_actions = env.available_actions.copy()
#                 with torch.no_grad():
#                     state_tensor = torch.tensor(state, device=device)
#                     actor_output = actor(state_tensor)
#                     mask = build_action_mask(env.action_size, available_actions, device)
#                     dist = Categorical(logits=actor_output + mask)
#                     action = dist.sample().item()
#                     next_state, reward, done, won, attempts = env.step(action)
#
#                 episodes[episode].append({
#                     'state': state,
#                     'next_state': next_state.copy(),
#                     'action': action,
#                     'reward': float(reward),
#                     'done': done,
#                     'available_actions': available_actions,
#                 })
#
#         return episodes
#
#     # Protagonist episode
#     trajectories_p = collect_agent_episode(actor_p, batch_size=10)
#     rewards_p = [[t['reward'] for t in episode] for episode in trajectories_p]
#
#     env.reset()
#     env.target_word = target
#     # Antagonist episode
#     trajectories_a = collect_agent_episode(actor_a, batch_size=10)
#
#     rewards_a = [[t['reward'] for t in episode] for episode in  trajectories_a]
#     regret = [sum(ra) - sum(rp) for ra, rp in zip(rewards_a, rewards_p)]
#     print(regret)
#     return {
#         'trajectories_p': trajectories_p,
#         'trajectories_a': trajectories_a,
#         'regret': regret,
#     }
#
# def compute_and_apply_gradients(
#         episodes,
#         action_size,
#         actor_p,
#         critic_p,
#         actor_a,
#         critic_a,
#         optimizer_actor_p,
#         optimizer_critic_p,
#         optimizer_actor_a,
#         optimizer_critic_a,
#         gamma=0.99, # Discount factor
# ):
#     """
#     Computes and applies gradients for both agents and all workers based on the collected trajectory data.
#     """
#     torch.set_num_threads(12)
#
#     for episode in episodes["trajectories_p"]:
#         optimizer_actor_p.zero_grad()
#         optimizer_critic_p.zero_grad()
#         batch_actor_loss_p = torch.zeros((), device=device)
#         batch_critic_loss_p = torch.zeros((), device=device)
#         steps_p = 0
#         for t in episode:
#             state = torch.tensor(t['state'], device=device)
#             next_state = torch.tensor(t['next_state'], device=device)
#             action = torch.tensor(t['action'], device=device, dtype=torch.long)
#             reward = torch.tensor(t['reward'], device=device)
#             done = torch.tensor(float(t['done']), device=device)
#
#             actor_output = actor_p(state)
#             value = critic_p(state).squeeze()
#             next_value = critic_p(next_state).squeeze()
#             mask = build_action_mask(action_size, t['available_actions'], device)
#             dist = Categorical(logits=actor_output + mask)
#             log_prob = dist.log_prob(action).squeeze()
#             td_error = reward + gamma * next_value * (1 - done) - value
#
#             batch_actor_loss_p -= log_prob * td_error.detach()
#             batch_critic_loss_p += td_error.pow(2)
#             steps_p += 1
#
#         batch_actor_loss_p /= max(steps_p, 1)
#         batch_critic_loss_p /= max(steps_p, 1)
#
#         batch_actor_loss_p.backward()
#         optimizer_actor_p.step()
#
#         batch_critic_loss_p.backward()
#         optimizer_critic_p.step()
#
#     for episode in episodes['trajectories_a']:
#         optimizer_actor_a.zero_grad()
#         optimizer_critic_a.zero_grad()
#         batch_actor_loss_a = torch.zeros((), device=device)
#         batch_critic_loss_a = torch.zeros((), device=device)
#         steps_a = 0
#         for t in episode:
#             state = torch.tensor(t['state'], device=device)
#             next_state = torch.tensor(t['next_state'], device=device)
#             action = torch.tensor(t['action'], device=device, dtype=torch.long)
#             reward = torch.tensor(t['reward'], device=device)
#             done = torch.tensor(float(t['done']), device=device)
#
#             actor_output = actor_a(state)
#             value = critic_a(state).squeeze()
#             next_value = critic_a(next_state).squeeze()
#             mask = build_action_mask(action_size, t['available_actions'], device)
#             dist = Categorical(logits=actor_output + mask)
#             log_prob = dist.log_prob(action).squeeze()
#             td_error = reward + gamma * next_value * (1 - done) - value
#
#             batch_actor_loss_a -= log_prob * td_error.detach()
#             batch_critic_loss_a += td_error.pow(2)
#             steps_a += 1
#
#
#         batch_actor_loss_a /= max(steps_a, 1)
#         batch_critic_loss_a /= max(steps_a, 1)
#
#         batch_actor_loss_a.backward()
#         optimizer_actor_a.step()
#         batch_critic_loss_a.backward()
#         optimizer_critic_a.step()
#
#
# def train(
#         num_episodes=1000,
#         protagonist_mode="easy",
#         antagonist_mode="easy",
#         vocab_path="src/data/answers.txt",
#         alpha=0.01, # Learning rate for the adversary
#         epsilon=0.2, # Exploration
#         actor_lr = 0.01,
#         critic_lr = 0.01,
# ):
#     np.random.seed(439)
#     torch.manual_seed(439)
#
#     env = WordleEnv(mode=protagonist_mode)
#
#     actor_p = Actor(env.state_size, env.action_size).to(device)
#     critic_p = Critic(env.state_size).to(device)
#
#     actor_a = Actor(env.state_size, env.action_size).to(device)
#     critic_a = Critic(env.state_size).to(device)
#
#     # actor_p.share_memory()
#     # critic_p.share_memory()
#     # actor_a.share_memory()
#     # critic_a.share_memory()
#
#     optimizer_actor_p = optim.Adam(actor_p.parameters(), lr=actor_lr)
#     optimizer_critic_p = optim.Adam(critic_p.parameters(), lr=critic_lr)
#
#     optimizer_actor_a = optim.Adam(actor_a.parameters(), lr=actor_lr)
#     optimizer_critic_a = optim.Adam(critic_a.parameters(), lr=critic_lr)
#
#     adversary = Adversary(vocab_path=vocab_path, alpha=alpha, epsilon=epsilon)
#
#     completed_episodes = 0
#     # with mp.Pool(processes=num_workers) as pool:
#         while completed_episodes < num_episodes:
#             # batch_size = min(num_workers, num_episodes - completed_episodes)
#             # jobs = []
#             # adversary_actions = []
#             # for worker in range(batch_size):
#             adversary_action = adversary.select_action()
#             target = env.vocab[adversary_action]
#                 # seed = worker + completed_episodes
#
#                 adversary_actions.append(adversary_action)
#                 # jobs.append(
#                 #     (
#                 #         seed,
#                 #         target,
#                 #         mode,
#                 #         actor_p,
#                 #         actor_a,
#                 #     )
#                 # )
#                 #
#                 # all_trajectories = pool.starmap(collect_episode_trajectories, jobs)
#
#             compute_and_apply_gradients(
#                 all_trajectories,
#                 env.action_size,
#                 actor_p,
#                 critic_p,
#                 actor_a,
#                 critic_a,
#                 optimizer_actor_p,
#                 optimizer_critic_p,
#                 optimizer_actor_a,
#                 optimizer_critic_a
#             )
#
#             for adversary_action, result in zip(adversary_actions, all_trajectories):
#                 adversary.observe(adversary_action, result['regret'])
#
#             completed_episodes += batch_size
#
#     return actor_p, critic_p, actor_a, critic_a, adversary
#
# if __name__ == "__main__":
#     # mp.set_start_method("spawn", force=True)
#     # num_episodes = 1000
#     # start = time.time()
#     # train(
#     #     num_episodes = num_episodes,
#     #     num_workers = 12,
#     #     mode = "easy",
#     #     vocab_path = "src/data/answers.txt",
#     #     alpha = 0.01,
#     #     epsilon = 0.2,
#     #     actor_lr = 0.01,
#     #     critic_lr = 0.01,
#     # )
#     # end = time.time()
#     # print(f"Training time: {end - start} seconds")