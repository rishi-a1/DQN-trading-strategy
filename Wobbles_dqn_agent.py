
# Importing modules and setting up
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from Ctrading_env import trading_env
import yfinance as yf
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Initialising the replay buffer class
class ReplayBuffer:
    # Making a queue that removes the oldest experiences as soon as the capacity is reached
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    # Adding a transition to the queue
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    # Taking a sample of the existing transitions in the replay buffer
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def _len(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 64)
        # 64 neurons condensing to 3 neurons to represent the q value for each action taken in the state
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # returning the Q-value
        return self.layer3(x)

# batch_size is the number of transitions sampled from the replay buffer
# gamma is the discount factor in the Markov Decision Process
# eps_init is the starting value of epsilon
# eps_end is the final value of epsilon
# eps_dec controls the rate of exponential decay of epsilon, higher means a slower decay
# update_rate is the update rate of the target network
# lr is the learning rate of the ``AdamW`` optimizer

batch_size = 128
gamma = 0.99
eps_init = 1
eps_end = 0.01
eps_dec = 2500
update_rate = 0.005
lr = 3e-4

# Data identifiers
ticker_symbol = "SPY"  # using standard S&P for data
start_date = "2020-01-01"
end_date = "2024-01-01"  # four-year learning period


# Download the historical data
data = yf.download(ticker_symbol, start=start_date, end=end_date, group_by='column', auto_adjust=False, multi_level_index=False)
price_data = data['Adj Close']
volume_data = data['Volume']

# Environment setup for agent us
env = trading_env(price_data, volume_data)

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# initialising the policy network and target network
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)

transition_capacity = 50000
memory = ReplayBuffer(transition_capacity)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_init - eps_end) * \
                    math.exp(-1. * steps_done / eps_dec)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if memory._len() < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# defining number of episodes used in learning
n_episodes = 100
for i_episode in range(n_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor(np.array([reward], dtype=np.float32), device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Optimize policy network
        optimize_model()

        # Soft update of the target network's weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * update_rate + target_net_state_dict[key] * (1 - update_rate)
        target_net.load_state_dict(target_net_state_dict)
        if done:
            episode_durations.append(t + 1)
            print(f"Episode: {i_episode}")
            print(f"Total reward (shaped): {env.total_reward:.4f}")
            print(f"Actual P&L (1 share):  ${env.portfolio_value:.2f}")
            print(f"Buy & Hold P&L (1 share): ${env.prices[-1] - env.prices[env.window_size]:.2f}")
            break

MODEL_SAVE_PATH = '/Networks/dqn_trading_policy_net.pth'

# Using policy_net variable to save its state_dict
torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)

print(f'Model saved to {MODEL_SAVE_PATH}')
print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
