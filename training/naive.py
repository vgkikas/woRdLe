import torch
import numpy as np
import torch.optim as optim
import time
from src.env.Wordle import WordleEnv
from src.models.ActorCritic import Actor, Critic
from src.utils.helpers import build_vocab_mask

# Setup device
device = torch.device("cpu") # Check README

start = time.time()


def train_improved(num_episodes=1000000, batch_size=4):
    subset = [
        "CIGAR", "REBUT", "SISSY", "HUMPH", "AWAKE", "BLUSH", "FOCAL", "EVADE", "NAVAL", "SERVE",
        "HEATH", "DWARF", "MODEL", "KARMA", "STINK", "GRADE", "QUIET", "BENCH", "ABATE", "FEIGN",
        "MAJOR", "DEATH", "FRESH", "CRUST", "STOOL", "COLON", "ABASE", "MARRY", "REACT", "BATTY",
        "PRIDE", "FLOSS", "HELIX", "CROAK", "STAFF", "PAPER", "UNFED", "WHELP", "TRAWL", "OUTDO",
        "ADOBE", "CRAZY", "SOWER", "REPAY", "DIGIT", "CRATE", "CLUCK", "SPIKE", "MIMIC", "POUND"
    ]
    env = WordleEnv(mode="easy")
    subset = env.words.copy()
    vocab = env.vocab.copy()
    env.words = subset[:20]
    env.vocab = subset[:20]
    env.action_size = len(env.words)

    # env.vocab = env.words
    ohe_matrix = np.zeros((26 * env.word_length, env.action_size))
    for i, word in enumerate(env.words):
        for pos, char in enumerate(word):
            letter_idx = ord(char) - 65
            ohe_matrix[pos * 26 + letter_idx, i] = 1
    env.ohe_matrix = ohe_matrix

    actor = Actor(env.state_size, env.ohe_matrix).to(device)
    critic = Critic(env.state_size).to(device)

    opt_actor = optim.Adam(actor.parameters(), lr=0.001)
    opt_critic = optim.Adam(critic.parameters(), lr=0.001)

    reward_history = []
    win_history = []

    trajectories = []

    print(f"Starting improved test with {len(subset)} words and batch size {batch_size}.")

    for i in range(1, num_episodes + 1):
        env.reset()
        done = False
        ep_reward = 0

        if i == 2000:
            env.words = subset[:100]
            env.vocab = subset[:100]
            env.action_size = len(env.words)
            ohe_matrix = np.zeros((26 * env.word_length, env.action_size))
            for i, word in enumerate(env.words):
                for pos, char in enumerate(word):
                    letter_idx = ord(char) - 65
                    ohe_matrix[pos * 26 + letter_idx, i] = 1
            env.ohe_matrix = ohe_matrix
            actor.ohe_matrix = torch.tensor(ohe_matrix, device=device, dtype=torch.float32)
        elif i == 20000:
            env.words = subset[:500]
            env.vocab = subset[:500]
            env.action_size = len(env.words)
            ohe_matrix = np.zeros((26 * env.word_length, env.action_size))
            for i, word in enumerate(env.words):
                for pos, char in enumerate(word):
                    letter_idx = ord(char) - 65
                    ohe_matrix[pos * 26 + letter_idx, i] = 1
            env.ohe_matrix = ohe_matrix
            actor.ohe_matrix = torch.tensor(ohe_matrix, device=device, dtype=torch.float32)
        elif i == 40000:
            env.words = subset
            env.vocab = vocab
            env.action_size = len(env.words)
            ohe_matrix = np.zeros((26 * env.word_length, env.action_size))
            for i, word in enumerate(env.words):
                for pos, char in enumerate(word):
                    letter_idx = ord(char) - 65
                    ohe_matrix[pos * 26 + letter_idx, i] = 1
            env.ohe_matrix = ohe_matrix
            actor.ohe_matrix = torch.tensor(ohe_matrix, device=device, dtype=torch.float32)
        while not done:
            state = env.current_state.copy()
            available_actions = env.available_actions.copy()
            state_tensor = torch.tensor(state, device=device, dtype=torch.float32)

            with torch.no_grad():
                actor_output = actor(state_tensor)
                mask = build_vocab_mask(env.action_size, available_actions, device)
                dist = actor.get_distribution(actor_output, mask=mask)
                action = dist.sample().item()

            next_state, reward, done, won, attempts = env.step(action)

            trajectories.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state.copy(),
                'done': done,
                'available_actions': available_actions
            })
            ep_reward += reward

        if i % batch_size == 0:
            # Update Agent
            s = torch.tensor(np.array([t['state'] for t in trajectories]), device=device, dtype=torch.float32)
            a = torch.tensor([t['action'] for t in trajectories], device=device)
            r = torch.tensor([t['reward'] for t in trajectories], device=device, dtype=torch.float32)
            ns = torch.tensor(np.array([t['next_state'] for t in trajectories]), device=device, dtype=torch.float32)
            d = torch.tensor([float(t['done']) for t in trajectories], device=device, dtype=torch.float32)

            # Critic update
            curr_v = critic(s).squeeze(-1)
            next_v = critic(ns).squeeze(-1)
            target = r + 0.99 * next_v * (1 - d)
            td_error = target - curr_v

            opt_critic.zero_grad()
            td_error.pow(2).mean().backward()
            opt_critic.step()

            # Actor update with Advantage Normalization
            advantage = td_error.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)

            # We need to re-calculate distributions for the batch
            actor_outputs = actor(s)
            # Re-calculating masks for the batch is complex, so let's simplify and just use the logged actions
            # Or use a loop for the actor update to handle masks correctly
            actor_loss = 0
            for idx in range(len(trajectories)):
                t = trajectories[idx]
                mask = build_vocab_mask(env.action_size, t['available_actions'], device)
                dist = actor.get_distribution(actor(torch.tensor(t['state'], device=device, dtype=torch.float32)),
                                              mask=mask)
                log_prob = dist.log_prob(torch.tensor(t['action'], device=device))
                actor_loss += -(log_prob * advantage[idx]) - 0.1 * dist.entropy()

            actor_loss = actor_loss / len(trajectories)
            opt_actor.zero_grad()
            actor_loss.backward()
            opt_actor.step()

            trajectories = []

        reward_history.append(ep_reward)
        win_history.append(1 if won else 0)

        if i % 200 == 0:
            avg_r = np.mean(reward_history[-200:])
            win_rate = np.mean(win_history[-200:]) * 100
            print(f"Episode {i:5d} | Avg Reward: {avg_r:6.2f} | Win Rate: {win_rate:5.1f}%")
            print(time.time() - start)


if __name__ == "__main__":
    train_improved()
