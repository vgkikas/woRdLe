import numpy as np

class WordleEnv:
    def __init__(self, word_length=5, max_attempts=6, global_dataset_path='src/data/wordle_actual.txt', target_dataset_path='src/data/answers.txt', mode=None):
        self.word_length = word_length
        self.max_attempts = max_attempts
        self.target_word = ''
        self.attempts_left = self.max_attempts
        self.attempts = 0
        self.current_guess = ''

        # Load global dataset
        with open(global_dataset_path, 'r', encoding='utf-8') as f:
            words = [word.strip().upper() for word in f.readlines() if len(word.strip()) == word_length]
            self.words = words

        # Load the curriculum/answers subset
        if target_dataset_path is not None:
            # Introduce graded difficulty by loading different datasets based on the mode, if specified. Otherwise, load the target dataset as the vocabulary.
            if mode is not None:
                with open("src/data/dataset_1_easy.txt", 'r', encoding='utf-8') as f:
                    vocab = [word.strip().upper() for word in f.readlines() if len(word.strip()) == word_length]
                    self.vocab = vocab
                with open("src/data/dataset_2_medium.txt", 'r', encoding='utf-8') as f:
                    vocab = [word.strip().upper() for word in f.readlines() if len(word.strip()) == word_length]
                    self.vocab.extend(vocab)
                with open("src/data/dataset_3_hard.txt", 'r', encoding='utf-8') as f:
                    vocab = [word.strip().upper() for word in f.readlines() if len(word.strip()) == word_length]
                    self.vocab.extend(vocab)
            else:
                with open(target_dataset_path, 'r', encoding='utf-8') as f:
                    vocab = [word.strip().upper() for word in f.readlines() if len(word.strip()) == word_length]
                self.vocab = vocab
        else:
            self.vocab = self.words

        # State space has 391 dimensions (3 for each letter (gray, yellow, and green states) for each of the 5 positions, plus the first state indicating how many guesses the agent has left
        self.state_size = 391
        # Possible actions are the number of words in the dataset
        self.action_size = 26 * self.word_length # One-hot encoding of the words
        self.available_actions = list(range(self.action_size))
        ohe_matrix = np.zeros((self.action_size, len(self.words)))  # Creates a one-hot encoding matrix for the entire dataset, to be used for extracting actual words from the actor's output
        for i, word in enumerate(self.words):
            for pos, char in enumerate(word):
                letter_idx = ord(char) - 65
                ohe_matrix[pos * 26 + letter_idx, i] = 1
        self.ohe_matrix = ohe_matrix
        # Current state starts as all zeros one hot encoded matrix, then it will be built after each move. First index indicates the number of attempts
        self.current_state = np.zeros(self.state_size, dtype=np.float32)
        self.current_state[0] = self.attempts_left

    def get_state(self):
        return self.current_state

    @staticmethod
    def get_feedback(guess, target):
        """
        Provides Wordle feedback (0 = Gray, 1 = Yellow, 2 = Green) for guess based on target.
        """
        feedback = [0] * len(guess)
        target_counts = {}
        
        for char in target:
            target_counts[char] = target_counts.get(char, 0) + 1
            
        for i, (g_char, t_char) in enumerate(zip(guess, target)):
            if g_char == t_char:
                feedback[i] = 2
                target_counts[g_char] -= 1
                
        for i, (g_char, t_char) in enumerate(zip(guess, target)):
            if feedback[i] == 0 and target_counts.get(g_char, 0) > 0: # If feedback is not green but the letter is still in the target word
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

    def reset(self):
        self.target_word = np.random.choice(self.vocab)
        self.attempts_left = self.max_attempts
        self.attempts = 0
        self.current_guess = '_' * self.word_length
        self.available_actions = list(range(self.action_size))
        self.current_state = np.zeros(self.state_size, dtype=np.float32)
        self.current_state[0] = self.attempts_left


    # Each time we make an action (make a guess), we check how many of the letters are correct
    def step(self, action):
        self.current_guess = self.words[action]
        self.attempts += 1
        done = False
        won = False
        if self.current_guess == self.target_word:
            reward = 10 * self.attempts_left # Incentivizes early guesses
            done = True
            won = True
        else:
            reward = sum(self.get_feedback(self.current_guess, self.target_word)) # Rewards correct letters, however not enough to encourage repeated guesses compared to winning
            self.attempts_left -= 1
            if self.attempts_left == 0:
                reward = (1-self.max_attempts)*10 # Negative reward for losing, big enough to incentivize exploration instead of trying the same word over and over again
                done = True
        self.update_state_from_feedback()

        return self.get_state(), reward, done, won

    # Printing output purposes.
    def render(self):
        print(f"Current guess: {self.current_guess}")
        print(f"Target word: {self.target_word}")
        print(f"Attempts left: {self.attempts_left}")
