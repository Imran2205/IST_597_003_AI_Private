import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import gymnasium
import miniwob
from miniwob.action import ActionTypes, ActionSpaceConfig


class ActionHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Action type prediction
        self.action_type = nn.Linear(hidden_dim, 3)  # Example: CLICK_COORDS, TYPE_TEXT, PRESS_KEY

        # Coordinate prediction (for click actions)
        self.coords = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # x, y coordinates
        )

        # Text prediction (for type actions)
        self.text = nn.Linear(hidden_dim, 128)  # Vocabulary size

        # Key prediction (for key press actions)
        self.key = nn.Linear(hidden_dim, 10)  # Number of allowed keys


class DQN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.action_head = ActionHead(hidden_dim)

    def forward(self, x):
        features = self.encoder(x)
        return {
            'action_type': self.action_head.action_type(features),
            'coords': self.action_head.coords(features),
            'text': self.action_head.text(features),
            'key': self.action_head.key(features)
        }


class DQLAgent:
    def __init__(self, environ, learning_rate=0.001, gamma=0.99, epsilon=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)

        self.gamma = gamma
        self.epsilon = epsilon

        # Action space configuration
        self.action_types = ['CLICK_COORDS', 'TYPE_TEXT', 'PRESS_KEY']
        # self.screen_width = 800
        # self.screen_height = 600
        self.screen_height = environ.observation_space["screenshot"].shape[0]
        self.screen_width = environ.observation_space["screenshot"].shape[1]
        self.coord_bins = (20, 20)  # Discretize coordinate space

    def preprocess_observation(self, observation):
        features = []
        for element in observation['dom_info']:
            element_features = [
                element.get('left', 0) / self.screen_width,
                element.get('top', 0) / self.screen_height,
                element.get('width', 0) / self.screen_width,
                element.get('height', 0) / self.screen_height,
                1.0 if element.get('tag') == 'button' else 0.0,
                1.0 if element.get('tag') == 'input' else 0.0
            ]
            features.extend(element_features)

        features = features[:128]
        features.extend([0] * (128 - len(features)))
        return torch.tensor(features, dtype=torch.float32)

    def discretize_coords(self, coords):
        x_bin = int(coords[0] * self.coord_bins[0])
        y_bin = int(coords[1] * self.coord_bins[1])
        return np.array([
            (x_bin + 0.5) * (self.screen_width / self.coord_bins[0]),
            (y_bin + 0.5) * (self.screen_height / self.coord_bins[1])
        ])

    def __call__(self, observation):
        if random.random() < self.epsilon:
            action_type = random.randint(0, len(self.action_types) - 1)
            action = {'action_type': action_type}

            if self.action_types[action_type] == 'CLICK_COORDS':
                action['coords'] = np.array([
                    random.random() * self.screen_width,
                    random.random() * self.screen_height
                ])
            elif self.action_types[action_type] == 'TYPE_TEXT':
                action['text'] = 'example'  # Simplified text generation
            elif self.action_types[action_type] == 'PRESS_KEY':
                action['key'] = random.randint(0, 9)

        else:
            state = self.preprocess_observation(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
                action_type = torch.argmax(q_values['action_type']).item()
                action = {'action_type': action_type}

                if self.action_types[action_type] == 'CLICK_COORDS':
                    coords = q_values['coords'][0].cpu().numpy()
                    action['coords'] = self.discretize_coords(coords)
                elif self.action_types[action_type] == 'TYPE_TEXT':
                    text_logits = q_values['text'][0]
                    # Convert logits to text (simplified)
                    action['text'] = 'example'
                elif self.action_types[action_type] == 'PRESS_KEY':
                    action['key'] = torch.argmax(q_values['key']).item()

        return action

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([self.preprocess_observation(s) for s in states]).to(self.device)
        next_states = torch.stack([self.preprocess_observation(s) for s in next_states]).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        current_q_values = self.q_network(states)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)

        # Calculate loss for each action component
        losses = []

        # Action type loss
        action_type_loss = nn.CrossEntropyLoss()(
            current_q_values['action_type'],
            torch.tensor([a['action_type'] for a in actions]).to(self.device)
        )
        losses.append(action_type_loss)

        # Add other losses for coords, text, and keys based on action types

        loss = sum(losses)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if random.random() < 0.01:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * 0.995)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


def train_agent(env, episodes=1000):
    agent = DQLAgent(environ=env)

    for episode in range(episodes):
        observation, info = env.reset()
        episode_reward = 0

        while True:
            action = agent(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)

            agent.store_transition(observation, action, reward, next_observation,
                                   terminated or truncated)
            agent.train()

            episode_reward += reward
            observation = next_observation

            if terminated or truncated:
                print(f"Episode {episode + 1}, Reward: {episode_reward}")
                break

    return agent


# Register all MiniWoB environments
gymnasium.register_envs(miniwob)

# Define action space config
action_config = ActionSpaceConfig(
    action_types=[
        ActionTypes.CLICK_COORDS,
        ActionTypes.TYPE_TEXT,
        ActionTypes.PRESS_KEY
    ],
    coord_bins=(20, 20),  # Discretize coordinate space
    text_max_len=10,
    allowed_keys=["<Enter>", "<Tab>"]
)

# Create environment with config
env = gymnasium.make(
    'miniwob/click-test-2-v1',
    render_mode='human',
    action_space_config=action_config
)

trained_agent = train_agent(env)

torch.save(trained_agent, 'saved_agent.pt')

# Load the trained agent
trained_agent = torch.load('saved_agent.pt')  # If you saved the model

# Test loop
episodes = 100
total_rewards = []

env.reset()
for episode in range(episodes):
    observation, info = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action = trained_agent(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    total_rewards.append(episode_reward)
    print(f"Episode {episode}: Reward = {episode_reward}")

print(f"Average Reward: {np.mean(total_rewards)}")
print(f"Success Rate: {sum(r > 0 for r in total_rewards) / episodes}")

# Save
# torch.save(trained_agent, 'saved_agent.pt')

# Load
# loaded_agent = torch.save('saved_agent.pt')