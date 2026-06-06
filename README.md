# woRdLe : A Deep Reinforcement Learning Approach
This is a mini-project for the course CS-439: Optimization for Machine Learning at EPFL.



##  Setup:
```shell
pip install -r requirements.txt
```
## Project Structure:
```
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── envs                            <- Environment directory
│   ├── models                          <- Model directory
│   ├── trainings                       <- Training directory
│   ├─  utils                           <- Utility directory
│
├── main.ipynb               <- Notebook with final results
│
├── requirements.txt        <- File for installing python dependencies
└── README.md
```
Environment:
State: A 391-dimensional vector encodes each letter's correctness and position relative to the target word.
Action: Selecting a word from a dictionary, with masking to prevent repeated guesses.
Reward System
Correct Word: +10 reward for guessing the target word correctly.
Attempts Exhausted: -10 penalty for exhausting attempts without guessing correctly.
Midway Progress: +1 reward for each correctly positioned letter in the guessed word.

Agents Applied :

1. Double Deep Q-Network:
DDQN employs two separate neural networks, known as the online network and the target network, to decouple the action selection from the value estimation. By periodically updating the target network with the parameters of the online network, DDQN stabilizes the learning process and improves performance, particularly in environments with large action spaces or complex dynamics.

2. Advantage Actor Critic:
In A2C, an actor network learns a policy to select actions, while a critic network estimates the value function to evaluate these actions. The advantage of A2C lies in its ability to update both the policy and the value function simultaneously, using the advantage function to guide learning. This approach leads to more stable training and faster convergence compared to traditional policy gradient methods.
The advantage function, A(s,a), measures the advantage of taking action a in state s​ over the expected value of the state under the current policy.
A(s,a) = Q(s,a) - V(s)

Used resources:

https://www.nytimes.com/games/wordle/index.html

https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

https://www.geeksforgeeks.org/actor-critic-algorithm-in-reinforcement-learning/



## A note on hardware optimization:
A significant amount of time was spent exploring the best hardware configuration for the project. Our testing across different hardware configurations (modern consumer-grade CPUs, older server-class CPUs, and GPUs) revealed that the performance bottleneck of the project was single-thread CPU performance. After trying to optimize the code for multi-threading, we found that the overhead of thread management outweighed the benefits. Similarly, due to the small number of low-dimensional layers in our neural networks, GPU utilization hindered performance. While it would be possible to further increase the dimensionality of the network's layers or the batch size, we believe that in a setting with many possible actions such as Wordle, the most important factor is experience. We therefore decided to keep the neural network minimal to facilitate faster learning per episode.

For the reasons stated above, we recommend using a CPU with high single-core performance for training. For peace of mind, the CPU provided by GitHub Codespaces is a good choice.

Team Members:
Farhan Ali, Jean Fregeville, Vasileios Gkikas