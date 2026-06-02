# woRdLe : A Deep Reinforcement Learning Approach

Team Members:
Harsh Shah, Anwit Damle, Ujjwal Agarwal

##  Setup:
```shell
pip install -r requirements.txt
```

Environment:
State: A 78-dimensional vector encodes each letter's correctness and position relative to the target word.
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

## Minimal training entrypoint (`main.py`)

The training flow from `main.ipynb` is now available as a script in `main.py`, with checkpoint resume for long runs.

List seed jobs:

```shell
python main.py --list-jobs --seeds 0,1,2,3
```

Run one seed locally:

```shell
python main.py --seeds 0,1 --job-index 0 --episodes-per-phase 100000
```

Run a tiny smoke test:

```shell
python main.py --seeds 0 --max-episodes 10 --checkpoint-every 5 --log-every 5
```

Checkpoint files are saved under `outputs/seed_<seed>/checkpoint_latest.pt` and resume automatically on restart.

For clusters, set one of these env vars per pod/task: `JOB_INDEX`, `SLURM_ARRAY_TASK_ID`, or `POD_INDEX`.

