# Executive Summary for Trick Takers 
(Deep Learning Bootcamp Summer 2024, The Erdos Institute)

## Team Members 
Shin Kim, Juergen Kritschgau, Sixuan Lou, Edward Varvak, Yizhen Zhao 

## Repository Structure
 - `trick_game/` importable Python package for the game environment, policies, DQN model, replay memory, training helpers, and evaluation helpers
   - `game.py` card game state and legal move logic
   - `env.py` RL-style environment wrapper
   - `policies.py` random, greedy, and neural-network policies
   - `models.py` DQN model architecture compatible with the saved weights
   - `replay.py` replay memory utilities
   - `dqn.py` DQN action selection, optimization, and target-network updates
   - `train_dqn.py` command-line DQN training entry point
   - `evaluate.py` command-line evaluation entry point
 - `utils/` compatibility shims for older notebooks that import `utils.card_engine` or `utils.greedy_agent`
 - `tests/` focused pytest coverage for game rules, encoded legal masks, environment stepping, model compatibility, and CLI parsing
 - `dqn_training with legal moves.ipynb` historical training notebook for a neural net forced to always play legal moves
 - `dqn_training.ipynb` historical training notebook for a neural net trying to learn the rules of Spades
 - `dqn_training with previous training.ipynb` historical training notebook for training against a previously trained network
 - `experiments.ipynb` historical policy evaluation notebook
 - `reports/technical_findings.md` local technical findings report extracted from saved notebook outputs, with annotated figures in `reports/assets/`
 - `weights/` saved neural-network checkpoints from training notebooks
 - `figures/` and `notes/` supporting images and notes
 - `depreciated notebookes/` deprecated notebook work

## Setup

This project now uses `uv` for its Python environment.

```bash
uv sync
```

Run the test suite:

```bash
uv run python -m pytest -q
```

Launch notebooks:

```bash
uv run jupyter lab
```

## Command-Line Usage

Evaluate a baseline policy:

```bash
uv run trick-game-evaluate --policy random --games 1000
uv run trick-game-evaluate --policy greedy --games 1000
```

Evaluate a trained network:

```bash
uv run trick-game-evaluate --policy network --weights weights/q_fn_816-100000.pth --games 5000
```

Train a DQN that always chooses from legal moves:

```bash
uv run trick-game-train-dqn --episodes 5000 --legal-only --output weights/dqn-final.pth
```

Train against a frozen previously trained opponent:

```bash
uv run trick-game-train-dqn \
  --episodes 5000 \
  --legal-only \
  --opponent-weights weights/ev_q_function_output_outside_folder.pth \
  --output weights/dqn-vs-frozen-opponent.pth
```

The notebooks preserve historical outputs and plots for reference, but new work should prefer the `trick_game` package and CLI entry points.

## Overview
In reinforcement learning problems, the agent learns how to maximize a numerical reward signal through direct interactions with the environment and without relying on a complete model of the environment. In fact, agents using model free methods learn from raw experience and without any inferences about how the environment will behave. An important model free method is the use of the Q-Learning algorithm to approximate the optimal action value function. However, it can be impractical to estimate the optimal action value function for every possible state-action pair. Deep Q-Learning uses a neural network trained with a variant of Q-Learning as a nonlinear function approximator of the optimal action value function. Our objective is to use Deep Q-Learning to train an agent to make legal moves and/or win tricks while playing the card game Spades (without bidding).

## Methodology and Results
### Creating the Environment
We created an environment that can randomly initialize and play through games, keep track of the states, and stop the game when the agent plays an illegal move. At each stage in the game, the game state contains information about the cards in the current player’s hand, the tricks each player took, the cards that have been played and the turns in which they were played, and the cards that have not been played. Each card is one-hot encoded as a vector of length 52 and the game state is a vector of length 3016. When the agent plays a card, the other players randomly select a legal move on their respective turns.

We allowed the agent to change the environment in one of two ways. The agent feeds the input vector of length 3016 to the neural network and receives a vector of length 52 consisting of the action values. Then, the agent either chooses an action with the maximum action value or chooses a legal action with the maximum action value. In the former case, the agent may choose an illegal move and the game terminates at the next state if the agent does so. In the latter case, the agent always makes legal moves and only needs to learn how to maximize the number of tricks taken. 
  
### Architecture of the Neural Network
The network consists of two hidden linear layers each of which has 128 nodes and is followed by a rectifier activation. The output is a vector of length 52 and each element of the vector is the estimated value of the action value function when playing the card of the corresponding index given the state of the game specified by the input vector.

### Implementing Replay Memory and Optimizing Parameters
The replay memory is a list of 13 double ended queues (or deques) each of length 10000. An element in the i-th deque is a tuple, called a transition, consisting of the state of the particular game at the agent’s i-th turn, the action that the agent took at that state, the reward the agent received for taking the action, and the state that followed the agent’s action. The agent selects an action based on an epsilon-greedy policy with a decaying epsilon. Each time the agent selects an action, the corresponding transition tuple is appended onto the appropriate deque. We use the optimizer AdamW to update the parameters. The optimizer uses a batch of 100 transitions that are randomly sampled from the deques that are sufficiently filled. This ensures that the transitions that the agent uses to learn are not serially correlated. Additionally, the stratification of the replay memory described above helps the agent learn from different stages of the game. In particular, this prevents the agent from only sampling transitions corresponding to the first few turns of the games which take up most of the replay memory in the early stages of training.

### Optimizing Hyperparameters 
One of the more sensitive hyperparameters was the discount rate (gamma). The gamma value was especially sensitive when the agent selected actions without regard to whether or not the action is legal. For gamma values close to 1, the values of the estimated action value functions tend to blow up in magnitude resulting in poor agent performance. For gamma values close to 0, the agent was able to learn the rules of the game very well resulting in an average duration of around 11 turns for 100 simulated games. To reach the balance between learning to play legal moves and looking further into the future to maximize the value function, we decided a discount rate around 0.3 to be optimal.    

## Results
In the case when the agent selected actions based on the network output alone, the agent was able to play for 8.5 turns on average in 100 simulated games. When the agent only selected legal actions, the agent was able to take 3.7 tricks on average in 1000 simulated games. To evaluate this performance, we implemented a one turn greedy algorithm where the agent chooses the card with the highest chance of winning the trick at each turn. This baseline model was able to take 3.8 tricks on average in 1000 simulated games.

Additional saved notebook findings have been extracted into `reports/technical_findings.md`. Highlights from the preserved outputs include:

 - random legal policy: 3.308 average reward per game over 1000 games
 - greedy policy: 3.8 average reward per game over 1000 games
 - trained checkpoint `weights/q_fn_816-100000.pth`: 3.819 average reward per game over 5000 games in one saved evaluation
 - trained checkpoint `weights/ev_q_function_output_outside_folder.pth`: 3.742 average reward per game over 5000 games
 - legal-action masking reliably completes games in the saved training/evaluation outputs, while unmasked legality learning is much more sensitive and often terminates early

## Future Directions 

So far, the agent played against other players that select a random legal move when it is their turn. It would be interesting to see how the agent learns when we allow for the other players to select actions based on a trained neural network.
Given that the inputs to the neural network consist entirely of one hot encoded information, we opted to use linear layers. We also kept the network depth fixed to keep the training time relatively short. We can experiment with the network architecture to find one that optimizes agent performance.

The refactor also makes it easier to explore:

 - Double DQN and Dueling DQN
 - prioritized replay and n-step returns
 - PPO or actor-critic methods with masked categorical actions
 - more systematic self-play or opponent-pool training
 - richer state representations using card/rank/suit embeddings


## References

 - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
 - Andrew Barto and Richard Sutton. Reinforcement Learning : An Introduction. MIT Press.
