import gymnasium
import miniwob
import numpy as np
from gymnasium import spaces
import time


# This is our policy
class GradualMovePolicy:
    def __init__(self, step_size=5):
        self.step_size = step_size
        self.target_pos = None
        self.origin = np.array([0, 0])
        self.current_pos = None
        self.target = None

    def get_target_position(self, dom_elements):
        for element in dom_elements:
            if self.target == 'circle':
                key = 'tag'
            else:
                key = 'text'
            if element[key].strip() == self.target:
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

        # print(window_width, window_height)

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

        for element in observation['dom_elements']:
            if self.target == 'circle':
                key = 'tag'
            else:
                key = 'text'
            if element[key].strip() == self.target:
                # print(element['text'])
                cursor_over_button = (
                        element['left'][0] + (element['width'][0] / 2) <= self.current_pos[0] <= element['left'][0] + element['width'][0] and
                        element['top'][0] + (element['height'][0] / 2) <= self.current_pos[1] <= element['top'][0] + element['height'][0]
                )
                if cursor_over_button:
                    print("aaa")
                    if self.target == 'circle':
                        self.target = 'Submit'
                        self.target_pos = self.get_target_position(observation['dom_elements'])
                    return {
                        "action_type": 2,  # CLICK_COORD
                        "coords": self.current_pos,
                        "key": 0
                    }


        return {
            "action_type": 1,  # Move_COORD
            "coords": self.current_pos,
            "key": 0
        }

    def __call__(self, observation):
        # Reading the fields tuple
        current_observation = dict(observation['fields'])
        # target = current_observation['target']

        # Extract target from fields
        dom_elements = observation['dom_elements']

        self.initialize_position(observation)

        if self.target_pos is None:
            self.target = 'circle'
            self.target_pos = self.get_target_position(dom_elements)

        return self.move_toward_target()


# This is the main entry
# env = gymnasium.make('miniwob/click-test-2-v1', render_mode='human')
# env = gymnasium.make('miniwob/click-test-2-v1', render_mode='human')
# circle-center
env = gymnasium.make('miniwob/circle-center', render_mode='human')
# env.unwrapped.instance = env.unwrapped._hard_reset_instance()
env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=99999999999)

# Create our modified environment
# env = TimeBasedRewardWrapper(env)

try:
    # use our policy
    policy = GradualMovePolicy(step_size=5)
    observation, info = env.reset(seed=42)

    # show the target
    # assert observation["utterance"] == "Click button ONE."
    # assert observation["fields"] == (("target", "ONE"),)

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