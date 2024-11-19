import gymnasium
import miniwob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random

# Experience replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DistanceRewardWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_distance = None

    def get_target_position(self, dom_elements, target_text):
        """Find the target button's center position"""
        for element in dom_elements:
            if element['text'].strip() == target_text.strip():
                return np.array([
                    element['left'] + element['width'] / 2,
                    element['top'] + element['height'] / 2
                ])
        return None

    def calculate_distance(self, current_pos, target_pos):
        """Calculate Euclidean distance between current position and target"""
        return np.linalg.norm(current_pos - target_pos)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        target = dict(observation['fields'])['target']
        target_pos = self.get_target_position(
            observation['dom_elements'],
            target
        )
        # Initialize previous_distance with distance from (0,0)
        self.previous_distance = self.calculate_distance(
            np.array([0, 0]),
            target_pos
        )
        return observation, info

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)

        # Get current cursor position from action
        current_pos = action['coords']

        # Get target position
        target = dict(observation['fields'])['target']
        target_pos = self.get_target_position(
            observation['dom_elements'],
            target
        )

        # print(target_pos)

        # Calculate current distance
        current_distance = self.calculate_distance(current_pos, target_pos)

        # Calculate reward based on distance change
        if self.previous_distance is not None:
            if current_distance < self.previous_distance:
                reward = 1.0  # Distance decreased
            elif current_distance > self.previous_distance:
                reward = -1.0  # Distance increased
            else:
                reward = 0.0  # Distance unchanged
        else:
            reward = 0.0

        # Update previous distance
        self.previous_distance = current_distance

        # Episode terminates when cursor is over button
        for element in observation['dom_elements']:
            if element['text'].strip() == target.strip():
                # print(element['left'] , current_pos , element['left'] , element['width'], action['coords'])
                # print(element['top'] , current_pos[1] , element['top'] , element['height'])
                cursor_over_button = (
                        element['left'] <= current_pos[0] <= element['left'] + element['width'] and
                        element['top'] <= current_pos[1] <= element['top'] + element['height']
                )
                if cursor_over_button:
                    terminated = True
                    reward = 10.0  # Bonus reward for reaching the target
                break

        return observation, reward, terminated, truncated, info
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        return Experience(*zip(*experiences))

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.action_dim = action_dim

        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayBuffer()
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.step_counter = 0

        # Action space discretization
        self.step_size = 10  # Pixels per step
        self.actions = [
            (-self.step_size, 0),  # Left
            (self.step_size, 0),  # Right
            (0, -self.step_size),  # Up
            (0, self.step_size),  # Down
            (-self.step_size, -self.step_size),  # Diagonal
            (-self.step_size, self.step_size),
            (self.step_size, -self.step_size),
            (self.step_size, self.step_size)
        ]

    def get_state(self, observation, current_pos):
        # Extract target position
        target = dict(observation['fields'])['target']
        target_pos = None

        # Find target button position
        for element in observation['dom_elements']:
            if element['text'].strip() == target.strip():
                target_pos = np.array([
                    element['left'][0] + element['width'][0] / 2,
                    element['top'][0] + element['height'][0] / 2
                ])
                break

        if target_pos is None:
            return None

        # State: [current_x, current_y, target_x, target_y]
        # print(current_pos, target_pos, element['left'] , element['width'] )
        state = np.concatenate([current_pos, target_pos])
        return torch.FloatTensor(state).to(self.device)

    def select_action(self, state):
        if random.random() < self.epsilon:
            action_idx = random.randrange(len(self.actions))
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action_idx = q_values.max(0)[1].item()

        movement = self.actions[action_idx]
        return action_idx, movement

    def update_current_pos(self, current_pos, movement):
        new_pos = current_pos + np.array(movement)
        # Clip to ensure cursor stays within screen bounds
        new_pos = np.clip(new_pos, [0, 0], [160, 160])  # Adjust bounds as needed
        return new_pos

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        experiences = self.memory.sample(self.batch_size)

        states = torch.stack(experiences.state)
        next_states = torch.stack(experiences.next_state)
        actions = torch.tensor(experiences.action, device=self.device)
        rewards = torch.tensor(experiences.reward, device=self.device, dtype=torch.float32)
        dones = torch.tensor(experiences.done, device=self.device, dtype=torch.float32)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_counter += 1
        if self.step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent():
    env = gymnasium.make('miniwob/click-test-2-v1', render_mode='human')
    env = DistanceRewardWrapper(env)

    state_dim = 4  # [current_x, current_y, target_x, target_y]
    action_dim = 8  # 8 possible movements
    agent = DQNAgent(state_dim, action_dim)

    current_pos = np.array([0, 0])
    episodes = 3

    try:
        for episode in range(episodes):
            observation, info = env.reset()
            state = agent.get_state(observation, current_pos)
            episode_reward = 0

            while True:
                action_idx, movement = agent.select_action(state)
                current_pos = agent.update_current_pos(current_pos, movement)

                # Convert to environment action format
                env_action = {
                    "action_type": 0,
                    "coords": current_pos,
                    "key": 0
                }

                next_observation, reward, terminated, truncated, info = env.step(env_action)
                next_state = agent.get_state(next_observation, current_pos)
                done = terminated or truncated

                # Store experience
                agent.memory.push(state, action_idx, reward, next_state, done)

                # Train agent
                agent.train()

                state = next_state
                episode_reward += reward

                if done:
                    print(f"Episode {episode + 1}, Total Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}")
                    current_pos = np.array([0, 0])
                    break

    finally:
        env.close()


if __name__ == "__main__":
    train_agent()