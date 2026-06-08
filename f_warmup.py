import json
import torch
import numpy as np
import torch.optim as optim
from src.env.Wordle import WordleEnv
from src.models.ActorCritic import Actor, Critic

device = torch.device("cpu") # Check README
torch.set_default_device(device)

def train_with_warmup(num_episodes=500000, batch_size=4):
    np.random.seed(439)
    torch.manual_seed(439)
    env = WordleEnv()
    full_words = env.words.copy()
    full_vocab = env.vocab.copy()
    subset=[]
    
    with open("src/data/dataset_1_easy.txt", 'r', encoding='utf-8') as f:
        easy_vocab = [word.strip().upper() for word in f.readlines() if len(word.strip()) == env.word_length]
        subset.extend(np.random.choice(easy_vocab, 20))
    with open("src/data/dataset_2_medium.txt", 'r', encoding='utf-8') as f:
        medium_vocab = [word.strip().upper() for word in f.readlines() if len(word.strip()) == env.word_length]
        subset.extend(np.random.choice(medium_vocab, 80))
    with open("src/data/dataset_3_hard.txt", 'r', encoding='utf-8') as f:
        hard_vocab = [word.strip().upper() for word in f.readlines() if len(word.strip()) == env.word_length]
        subset.extend(np.random.choice(hard_vocab, 400))

    actor = Actor(env.state_size, env.ohe_matrix).to(device)
    critic = Critic(env.state_size).to(device)

    optim_actor = optim.Adam(actor.parameters(), lr=0.001)
    optim_critic = optim.Adam(critic.parameters(), lr=0.001)

    def set_difficulty(words, vocab):
        """
        Sets the difficulty of the environment by changing the word list, vocabulary, and OHE matrix for the actor.
        """
        env.words = words
        env.vocab = vocab
        env.action_size = len(env.words)
        ohe_matrix = np.zeros((26 * env.word_length, env.action_size))
        for i, word in enumerate(env.words):
            for pos, char in enumerate(word):
                letter_idx = ord(char) - 65
                ohe_matrix[pos * 26 + letter_idx, i] = 1
        env.ohe_matrix = ohe_matrix
        actor.ohe_matrix = torch.tensor(ohe_matrix, device=device, dtype=torch.float32)

    # First level with 20 words
    set_difficulty(subset[:20], subset[:20])

    reward_history = []
    win_history = []
    attempt_history = []
    trajectories = []

    for i in range(1, num_episodes + 1):
        env.reset()
        done = False
        episode_reward = 0
        won=0
        if i == 2001: # First increase of difficulty
            set_difficulty(subset[:100], subset[:100])
        elif i == 20001: #  Second increase of difficulty
            set_difficulty(subset, subset)
        elif i == 40001: # Full datasets
            set_difficulty(full_words, full_vocab)
        episode_length = 0

        while not done:
            state = env.current_state.copy()
            available_actions = env.available_actions.copy()
            state_tensor = torch.tensor(state, device=device, dtype=torch.float32)

            with torch.no_grad():
                actor_output = actor(state_tensor)
                dist = actor.get_distribution(actor_output)
                action = dist.sample().item()

            next_state, reward, done, won = env.step(action)

            trajectories.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state.copy(),
                'done': done,
                'available_actions': available_actions
            })
            episode_reward += reward
            episode_length +=1

        if i % batch_size == 0:
            state = torch.tensor(np.array([t['state'] for t in trajectories]), device=device, dtype=torch.float32)
            action = torch.tensor([t['action'] for t in trajectories], device=device)
            reward = torch.tensor([t['reward'] for t in trajectories], device=device, dtype=torch.float32)
            next_state = torch.tensor(np.array([t['next_state'] for t in trajectories]), device=device, dtype=torch.float32)
            done = torch.tensor([float(t['done']) for t in trajectories], device=device, dtype=torch.float32)

            # Critic update
            current_value = critic(state).squeeze(-1)
            next_value = critic(next_state).squeeze(-1)
            td_error= reward + 0.99 * next_value * (1 - done) - current_value

            optim_critic.zero_grad()
            td_error.pow(2).mean().backward()
            optim_critic.step()

            # Actor update
            advantage = td_error.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9) # Batch normalization
            actor_loss = 0

            for idx in range(len(trajectories)):
                t = trajectories[idx]
                dist = actor.get_distribution(actor(torch.tensor(t['state'], device=device, dtype=torch.float32)))
                log_prob = dist.log_prob(torch.tensor(t['action'], device=device))
                actor_loss += -(log_prob * advantage[idx]) - 0.1 * dist.entropy() # Regularizing with entropy to encourage exploration

            actor_loss /= len(trajectories)
            optim_actor.zero_grad()
            actor_loss.backward()
            optim_actor.step()
            trajectories = []

        reward_history.append(episode_reward)
        win_history.append(int(won))
        attempt_history.append(episode_length)

        if i % 100000 == 0:
            with open(f'results/warmup/reward_history_{i}.json', 'w') as f:
                json.dump(reward_history, f)
            with open (f'results/warmup/win_history_{i}.json', 'w') as f:
                json.dump(win_history, f)
            with open (f'results/warmup/attempt_history_{i}.json', 'w') as f:
                json.dump(attempt_history, f)
            print(f"Saved data at {i}-th episode.")

if __name__ == "__main__":
    train_with_warmup()
