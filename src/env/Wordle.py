import numpy as np

class WordleEnv:
    def __init__(self, word_length=5, max_attempts=6, global_dataset_path='src/data/wordle_actual.txt', target_dataset_path='src/data/answers.txt', mode="easy"):
        self.word_length = word_length
        self.max_attempts = max_attempts
        self.mode = mode
        self.target_word = ''
        self.attempts_left = self.max_attempts
        self.attempts = 0
        self.current_guess = ''

        # Load global dataset
        with open(global_dataset_path, 'r', encoding='utf-8') as f:
            words = [word.strip().upper() for word in f.readlines() if len(word.strip()) == word_length]
            self.words = words

        # Load the subset for the teacher
        if target_dataset_path is not None:
            with open(target_dataset_path, 'r', encoding='utf-8') as f:
                vocab = [word.strip().upper() for word in f.readlines() if len(word.strip()) == word_length]
                self.vocab = vocab
        else:
            self.vocab = self.words

        # State space has 391 dimensions (3 for each letter (gray, yellow, and green states) for each of the 5 positions, plus the first state indicating how many guesses the agent has left)
        self.state_size = 391
        # Possible actions are the number of words in the dataset
        self.action_size = len(self.words)
        self.available_actions = list(range(self.action_size))
        # Current state starts as all zeros one hot encoded matrix, then it will be built after each move
        self.current_state = np.zeros(self.state_size, dtype=np.float32)
        self.current_state[0] = self.attempts_left

    def get_state(self):
        return self.current_state

    @staticmethod
    def get_feedback(guess, target):
        """Provides Wordle feedback (0 = Gray, 1 = Yellow, 2 = Green)"""
        feedback = [0] * len(guess)
        target_counts = {}
        
        for char in target:
            target_counts[char] = target_counts.get(char, 0) + 1
            
        for i, (g_char, t_char) in enumerate(zip(guess, target)):
            if g_char == t_char:
                feedback[i] = 2
                target_counts[g_char] -= 1
                
        for i, (g_char, t_char) in enumerate(zip(guess, target)):
            if feedback[i] == 0 and target_counts.get(g_char, 0) > 0:
                feedback[i] = 1
                target_counts[g_char] -= 1
                
        return feedback

    def update_state_from_feedback(self):
        self.current_state[0] = self.attempts_left
        feedback = self.get_feedback(self.current_guess, self.target_word)
        for pos, (char, fb) in enumerate(zip(self.current_guess, feedback)):
            if char == "_":
                continue
            letter_idx = ord(char) - 65
            self.current_state[1 + 78 * pos + fb * 26 + letter_idx] = 1
            # We have 3 feedback states (gray, yellow, green) for each of the 5 positions, and each letter can be one of 26 letters in the alphabet. So we have 78 (3*26) features for each position. We also have the first feature to indicate how many attempts are left, so we start from index 1.

    def remove_incompatible_words(self, current_guess):
        new_available_actions = []
        actual_feedback = self.get_feedback(current_guess, self.target_word)
        prev_available = self.available_actions.copy()
        for i in prev_available:
            candidate_word = self.words[i]
            simulated_feedback = self.get_feedback(current_guess, candidate_word)
            if simulated_feedback == actual_feedback:
                new_available_actions.append(i)

        self.available_actions = new_available_actions

    # This function masks action for the incompatible actions.
    def mask_action(self, action):
        if action in self.available_actions:
            self.available_actions.remove(action)
    
    # This function chooses a random number between 0 and length of dataset -1, which will be transformed into a word based on the index.
    def get_random_action(self):
        return np.random.randint(0, self.action_size)

    # Before starting each episode, the environment is reset to give the initial conditions.
    def reset(self):
        self.target_word = np.random.choice(self.vocab)
        self.attempts_left = self.max_attempts
        self.attempts = 0
        self.current_guess = '_' * self.word_length
        self.available_actions = list(range(self.action_size))
        self.current_state = np.zeros(self.state_size, dtype=np.float32)
        self.current_state[0] = self.attempts_left


    # Each time we make an action (make a guess), we check how many of the letters are correct.
    def step(self, action):
        self.current_guess = self.words[action]
        self.mask_action(action)  # Mask the taken action
        self.attempts += 1

        done = False
        won = False
        
        if self.current_guess == self.target_word:
            reward = 10 * self.attempts_left
            done = True
            won = True
        else:
            reward = sum(self.get_feedback(self.current_guess, self.target_word))
            self.attempts_left -= 1
            if self.attempts_left == 0:
                reward = (1-self.max_attempts)*10 # Negative reward for losing, big enough to incentivize exploration instead of trying the same word over and over again.
                done = True
        if self.mode == "hard": # Hard mode: remove incompatible words from the available actions
            self.remove_incompatible_words(self.current_guess)
        self.update_state_from_feedback()

        # modified: get the "won" info
        return self.get_state(), reward, done, won, self.attempts



    # Printing output purposes.
    def render(self):
        print(f"Current guess: {self.current_guess}")
        print(f"Target word: {self.target_word}")
        print(f"Attempts left: {self.attempts_left}")
