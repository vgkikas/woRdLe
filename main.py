import torch
import numpy as np
import torch.optim as optim
# import torch.multiprocessing as mp
import time
# import os
# from collections import namedtuple
# from itertools import count

from torch.distributions import Categorical

from src.env.Wordle import WordleEnv
from src.models.ActorCritic import Actor, Critic
from src.models.Adversary import Adversary
from src.utils.helpers import build_vocab_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


def select_action_actor(actor, state, available_actions, action_size):
    # Create a mask tensor for previously chosen actions
    actor_output = actor(state)
    mask = build_vocab_mask(action_size, available_actions, device)
    dist = actor.get_distribution(actor_output)
    action = dist.sample()
    # Add the mask to the actor output and sample from the distribution
    return action.item()

def collect_episode_trajectories(
        target,
        mode,
        actor_p,
        actor_a,
):
    """
    Runs one episode for the protagonist/antagonist and returns experience data.
    """
    env = WordleEnv(mode=mode)
    env.target_word = target

    def collect_agent_episode(actor):
        episode = []
        done = False
        while not done:
            state = env.current_state.copy()
            available_actions = env.available_actions.copy()
            with torch.no_grad():
                state_tensor = torch.tensor(state, device=device)
                # mask = build_vocab_mask(env.action_size, available_actions, device)
                # dist = Categorical(logits=actor_output + mask)
                action= select_action_actor(actor, state_tensor, available_actions, env.action_size)
                next_state, reward, done, won, attempts = env.step(action)

            episode.append({
                'state': state,
                'next_state': next_state.copy(),
                'action': action,
                'reward': float(reward),
                'done': done,
                'available_actions': available_actions,
            })

        return episode

    # Protagonist episode
    trajectories_p = collect_agent_episode(actor_p)

    reward_p = sum(t['reward'] for t in trajectories_p)

    env.reset()
    env.target_word = target
    # Antagonist episode
    trajectories_a = collect_agent_episode(actor_a)

    reward_a = sum(t['reward'] for t in trajectories_a)
    regret = reward_a - reward_p
    return {
        'trajectories_p': trajectories_p,
        'trajectories_a': trajectories_a,
        'regret': regret,
        'reward_p': reward_p,
        'reward_a': reward_a,
    }

def compute_and_apply_gradients(
        trajectories,
        action_size,
        actor_p,
        critic_p,
        actor_a,
        critic_a,
        optimizer_actor_p,
        optimizer_critic_p,
        optimizer_actor_a,
        optimizer_critic_a,
        gamma=0.99, # Discount factor
):
    """
    Computes and applies gradients for both agents and all workers based on the collected trajectory data.
    """
    optimizer_actor_p.zero_grad()
    optimizer_critic_p.zero_grad()
    optimizer_actor_a.zero_grad()
    optimizer_critic_a.zero_grad()
    batch_actor_loss_p = torch.zeros((), device=device)
    batch_critic_loss_p = torch.zeros((), device=device)
    batch_actor_loss_a = torch.zeros((), device=device)
    batch_critic_loss_a = torch.zeros((), device=device)
    steps_p = 0
    steps_a = 0

    all_td_errors_p = []
    all_td_errors_a = []

    for t in trajectories['trajectories_p']:
        state = torch.tensor(t['state'], device=device)
        next_state = torch.tensor(t['next_state'], device=device)
        action_tensor = torch.tensor(t['action'], device=device)
        reward = torch.tensor(t['reward'], device=device)
        done = torch.tensor(float(t['done']), device=device)

        value = critic_p(state).squeeze()
        next_value = critic_p(next_state).squeeze()
        # mask = build_vocab_mask(action_size, t['available_actions'], device)
        td_error = reward + gamma * next_value * (1 - done) - value
        dist = actor_p.get_distribution(actor_p(state))
        log_prob = dist.log_prob(action_tensor)
        all_td_errors_p.append(td_error.item())
        batch_actor_loss_p -= log_prob * td_error.detach()
        batch_critic_loss_p += td_error.pow(2)
        steps_p += 1

    for t in trajectories['trajectories_a']:
        state = torch.tensor(t['state'], device=device)
        next_state = torch.tensor(t['next_state'], device=device)
        action = torch.tensor(t['action'], device=device, dtype=torch.long)
        reward = torch.tensor(t['reward'], device=device)
        done = torch.tensor(float(t['done']), device=device)

        value = critic_a(state).squeeze()
        next_value = critic_a(next_state).squeeze()
        # mask = build_vocab_mask(action_size, t['available_actions'], device)
        td_error = reward + gamma * next_value * (1 - done) - value
        dist = actor_a.get_distribution(actor_a(state))
        log_prob = dist.log_prob(action)
        all_td_errors_a.append(td_error.item())
        batch_actor_loss_a -= log_prob * td_error.detach()
        batch_critic_loss_a += td_error.pow(2)
        steps_a += 1

    mean_td_p = np.mean(all_td_errors_p) if all_td_errors_p else 0
    mean_td_a = np.mean(all_td_errors_a) if all_td_errors_a else 0
    print(f"[diag] mean_td_error_p={mean_td_p:.4f} mean_td_error_a={mean_td_a:.4f}", flush=True)

    batch_actor_loss_p /= max(steps_p, 1)
    batch_critic_loss_p /= max(steps_p, 1)
    batch_actor_loss_a /= max(steps_a, 1)
    batch_critic_loss_a /= max(steps_a, 1)

    batch_actor_loss_p.backward()
    optimizer_actor_p.step()
    batch_critic_loss_p.backward()
    optimizer_critic_p.step()

    batch_actor_loss_a.backward()
    optimizer_actor_a.step()
    batch_critic_loss_a.backward()
    optimizer_critic_a.step()

    return {
        "actor_loss_p": batch_actor_loss_p.item(),
        "critic_loss_p": batch_critic_loss_p.item(),
        "actor_loss_a": batch_actor_loss_a.item(),
        "critic_loss_a": batch_critic_loss_a.item(),
    }

def train(
        num_episodes=1000,
        num_workers = 12,
        mode="easy",
        vocab_path="src/data/answers.txt",
        alpha=0.01,
        epsilon=0.2,
        actor_lr = 0.001,
        critic_lr = 0.001,
):
    np.random.seed(439)
    torch.manual_seed(439)

    env = WordleEnv(mode=mode)

    actor_p = Actor(env.state_size, env.action_size, env.ohe_matrix).to(device)
    critic_p = Critic(env.state_size).to(device)

    actor_a = Actor(env.state_size, env.action_size, env.ohe_matrix).to(device)
    critic_a = Critic(env.state_size).to(device)

    actor_p.share_memory()
    critic_p.share_memory()
    actor_a.share_memory()
    critic_a.share_memory()

    optimizer_actor_p = optim.Adam(actor_p.parameters(), lr=actor_lr)
    optimizer_critic_p = optim.Adam(critic_p.parameters(), lr=critic_lr)

    optimizer_actor_a = optim.Adam(actor_a.parameters(), lr=actor_lr)
    optimizer_critic_a = optim.Adam(critic_a.parameters(), lr=critic_lr)

    adversary = Adversary(vocab_path=vocab_path, alpha=alpha, epsilon=epsilon)

    completed_episodes = 0
    reward_history_p = []
    reward_history_a = []
    loss_history_p = []
    loss_history_a = []
    window_size = 50

    while completed_episodes < num_episodes:
        adversary_action = adversary.select_action()
        target = env.vocab[adversary_action]

                adversary_actions.append(adversary_action)
                jobs.append(
                    (
                        seed,
                        target,
                        mode,
                        actor_p,
                        actor_a,
                    )
                )

        all_trajectories = collect_episode_trajectories(target, mode, actor_p, actor_a)
        losses = compute_and_apply_gradients(
            all_trajectories,
            env.action_size,
            actor_p,
            critic_p,
            actor_a,
            critic_a,
            optimizer_actor_p,
            optimizer_critic_p,
            optimizer_actor_a,
            optimizer_critic_a,
        )
        adversary.observe(adversary_action, all_trajectories["regret"])

        reward_history_p.append(all_trajectories["reward_p"])
        reward_history_a.append(all_trajectories["reward_a"])
        loss_history_p.append(losses["actor_loss_p"] + losses["critic_loss_p"])
        loss_history_a.append(losses["actor_loss_a"] + losses["critic_loss_a"])
        completed_episodes += 1

        if completed_episodes % window_size == 0:
            avg_p = np.mean(reward_history_p[-window_size:])
            avg_a = np.mean(reward_history_a[-window_size:])
            avg_loss_p = np.mean(loss_history_p[-window_size:])
            avg_loss_a = np.mean(loss_history_a[-window_size:])
            print(
                f"ep={completed_episodes:5d} | "
                f"protag_r={avg_p:6.2f} protag_l={avg_loss_p:8.4f} | "
                f"antag_r={avg_a:6.2f} antag_l={avg_loss_a:8.4f}",
                flush=True,
            )
    return actor_p, critic_p, actor_a, critic_a, adversary


if __name__ == "__main__":
    print(device)
    num_episodes = 10000
    start = time.time()
    train(
        num_episodes = num_episodes,
        num_workers = 12,
        mode = "hard",
        vocab_path = "src/data/answers.txt",
        alpha = 0.01,
        epsilon = 0.2,
        actor_lr = 0.001,
        critic_lr = 0.001,
    )
    end = time.time()
    print(f"Training time: {end - start} seconds")