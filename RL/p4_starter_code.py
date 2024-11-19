import gymnasium
import miniwob
import numpy as np
from gymnasium import spaces
import time

#  This is the custom environment
class TimeBasedRewardWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_distance = None
        self.start_time = None
        # env.task_env.max_step_time = float('inf')  # Remove time limit
        self.step_rewards = []
        self.final_reward = 0.0
        self.reward_history = []

    def update_reward_display(self, reward):
        self.reward_history.append(reward)
        if len(self.reward_history) > 10:
            self.reward_history.pop(0)

        avg_reward = sum(self.reward_history) / len(self.reward_history)

        print(avg_reward, reward)

        exec_js = f"""
        (function() {{
            document.getElementById('reward-last').innerHTML = '{reward:.2f}';
            document.getElementById('reward-avg').innerHTML = '{avg_reward:.2f}';
        }})()
        """
        self.env.unwrapped.instance.driver.execute_script(exec_js)

    def get_target_position(self, dom_elements, target_text):
        for element in dom_elements:
            if element['text'].strip() == target_text.strip():
                # print(element['text'])
                return np.array([
                    element['left'][0] + element['width'][0] / 2,
                    element['top'][0] + element['height'][0] / 2
                ])
        return None

    def calculate_distance(self, current_pos, target_pos):
        return np.linalg.norm(current_pos - target_pos)

    def calculate_time_reward(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time <= 5:
            return 10 * (1 - elapsed_time / 5)  # Linear decrease from 10 to 0
        else:
            return -2 * (elapsed_time - 5)  # Linear decrease below 0

    # Overwriting the default reset function
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.start_time = time.time()
        target = dict(observation['fields'])['target']
        target_pos = self.get_target_position(
            observation['dom_elements'],
            target
        )
        # print(target_pos)
        self.previous_distance = self.calculate_distance(
            np.array([0, 0]),
            target_pos
        )
        self.step_rewards = []
        self.final_reward = 0.0
        return observation, info

    # Overwriting the default step function
    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        current_pos = action['coords']
        target = dict(observation['fields'])['target']
        target_pos = self.get_target_position(
            observation['dom_elements'],
            target
        )

        current_distance = self.calculate_distance(current_pos, target_pos)

        if self.previous_distance is not None:
            if current_distance < self.previous_distance:
                reward = 1.0
            elif current_distance > self.previous_distance:
                reward = -1.0
            else:
                reward = 0.0
        else:
            reward = 0.0

        self.step_rewards.append(reward)

        reward = np.mean(self.step_rewards)

        self.previous_distance = current_distance

        for element in observation['dom_elements']:
            if element['text'].strip() == target.strip():
                # print(element['text'])
                cursor_over_button = (
                        element['left'][0] + (element['width'][0] / 4) <= current_pos[0] <= element['left'][0] + element['width'][0] and
                        element['top'][0] + (element['height'][0] / 4) <= current_pos[1] <= element['top'][0] + element['height'][0]
                )
                if cursor_over_button:
                    terminated = True
                    reward_final = self.calculate_time_reward()
                    self.final_reward = reward_final
                    reward = (np.mean(self.step_rewards) + self.final_reward)
                    self.update_reward_display(reward)
                break

        return observation, reward, terminated, truncated, info


# This is our policy
class GradualMovePolicy:
    def __init__(self, step_size=5):
        self.step_size = step_size
        self.target_pos = None
        self.origin = np.array([0, 0])
        self.current_pos = None

    def get_target_position(self, dom_elements, target_text):
        for element in dom_elements:
            if element['text'].strip() == target_text.strip():
                # print(element['text'])
                return (
                    element['left'][0] + element['width'][0] / 2,
                    element['top'][0] + element['height'][0] / 2
                )
        return None

    def initialize_position(self, observation):
        if self.current_pos is None:
            self.current_pos = self.origin

    def move_toward_target(self):
        if self.target_pos is None or self.current_pos is None:
            return {
                "action_type": 1,
                "coords": np.array([0, 0]),
                "key": 0
            }

        window_height = env.observation_space["screenshot"].shape[0]
        window_width = env.observation_space["screenshot"].shape[1]

        current = np.array(self.current_pos)
        target = np.array(self.target_pos)

        direction = target - current
        distance = np.linalg.norm(direction)

        if distance > 0:
            direction = direction / distance * self.step_size
            noise = np.random.normal(0, self.step_size * 0.5, size=2)
            new_pos = current + direction + noise

            self.current_pos = np.clip(new_pos, [0, 0], [window_width, window_height])

            exec_js = f"""
            (function() {{
                var circle = document.createElement('div');
                circle.style.position = 'absolute';
                circle.style.left = '{self.current_pos[0]}px';
                circle.style.top = '{self.current_pos[1]}px';
                circle.style.width = '10px';
                circle.style.height = '10px';
                circle.style.backgroundColor = 'red';
                circle.style.borderRadius = '50%';
                circle.style.pointerEvents = 'none';
                circle.style.zIndex = '9999';
                document.body.appendChild(circle);
                setTimeout(() => circle.remove(), 70);
            }})()
            """
            env.unwrapped.instance.driver.execute_script(exec_js)

        # print(">>", current, direction[0], self.current_pos)

        return {
            "action_type": 1,  # CLICK_COORDS
            "coords": self.current_pos,
            "key": 0
        }

    def __call__(self, observation):
        # Reading the fields tuple
        current_observation = dict(observation['fields'])
        target = current_observation['target']

        # Extract target from fields
        dom_elements = observation['dom_elements']

        self.initialize_position(observation)

        if self.target_pos is None:
            self.target_pos = self.get_target_position(dom_elements, target)

        return self.move_toward_target()


# This is the main entry
env = gymnasium.make('miniwob/click-test-2-v1', render_mode='human')
env.unwrapped.instance = env.unwrapped._hard_reset_instance()
env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=99999999999)

# Create our modified environment
env = TimeBasedRewardWrapper(env)

try:
    # use our policy
    policy = GradualMovePolicy(step_size=5)
    observation, info = env.reset(seed=42)

    # show the target
    assert observation["utterance"] == "Click button ONE."
    assert observation["fields"] == (("target", "ONE"),)

    print(observation["utterance"], observation["fields"])
    
    final_reward = 0

    # run for some time
    for i in range(10000):

        action = policy(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        # print(f"Step {i}: Reward={reward}, Position={action['coords']}")

        if terminated or truncated:
            print(f"Episode finished with final reward: {reward}/11")
            time.sleep(3)
            policy.target_pos = None
            policy.current_pos = None
            observation, info = env.reset()
            print(observation["utterance"], observation["fields"])
finally:
    env.close()