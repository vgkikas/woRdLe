# Curriculum Learning for Wordle RL agents via Latent Action Space Exploitation
This is a mini-project for the course CS-439: Optimization for Machine Learning at EPFL.

##  Setup
```shell
pip install -r requirements.txt
```
## Project Structure
```
├── results                     <- Directory for saving results
│
│── src                         <- Source code
│   ├── data                    <- Data directory
│   ├── env                     <- Environment directory
│   ├── models                  <- Model directory
│
│── curriculum.py               <- File for training using a curriculum of increasing difficulty
│
│── naive.py                    <- File for training without curriculum
│
│── warmup.py                   <- File for training using a randomized curriculum
│
├── results.ipynb               <- Notebook with curriculum definition, final results and plots
│
├── requirements.txt            <- File for installing python dependencies
│
├── literature.bib              <- Literature references
│
└── README.md
```

## Project Description
This project implements a Reinforcemnt Learning agent to play the popular word-guessing game Wordle and explores pathways of training optimization via transfer learning.


### Environment

- State: A 391-dimensional vector encodes the number of attempts left and each letter's correctness and position relative to the target word.
- Latent action space: 130 dimensions.
- Action: The actor outputs are multiplied with an OHE matrix of all words in the vocabulary. The results are passed as logits to a categorical distribution and an action is sampled.
- Rewards: +2 for green, +1 for yellow, +10 $\times$ (1 + # of attempts left) for winning to encourage early winning.
- Penalty for losing: -50 to discourage exploitation of the reward system.


### Agents:
Advantage Actor Critic: An actor network that learns to maximize the expected return, and a critic network that estimates the value function to reduce variance in the policy gradient updates.

## Model Architecture:
- Actor: 391-dimensional input, one hidden layer of 256 units with ReLU activation, and a 130-dimensional output layer.
- Critic: The same architecture as the actor, but with a single output unit for the value function.
## A note on hardware optimization:
A significant amount of time was spent exploring the best hardware configuration for the project. Our testing across different hardware configurations (modern consumer-grade CPUs, older server-class CPUs, and GPUs) revealed that the performance bottleneck of the project was single-thread CPU performance and memory bandwidth. After trying to optimize the code for multi-threading, we found that the overhead of thread management outweighed the benefits. Similarly, due to the small number of low-dimensional layers in our neural networks, GPU utilization hindered performance. While it would be possible to further increase the dimensionality of the network's layers or the batch size, we believe that in a setting with many possible actions such as Wordle, the most important factor is experience. We therefore decided to keep the neural network minimal to facilitate faster learning per episode.

For the reasons stated above, we recommend using a modern CPU with high single-core performance for training.

## Team Members:
Farhan Ali, Jean Fregeville, Vasileios Gkikas
