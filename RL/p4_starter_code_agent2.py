import gymnasium
import miniwob
import numpy as np
from gymnasium import spaces
import time
from text_similarity_starter_code import compare_strings
from miniwob.action import ActionTypes


# This is our policy
class GradualMovePolicy:
    def __init__(self, step_size=5):
        self.origin = np.array([0, 0])
        self.target_ele_ind = 0
        self.submit_btn = None
        self.check_boxes = []
        self.check_boxes_texts = []
        self.match_matrix = None

    def __call__(self, observation_):
        # Reading the fields tuple
        prompt_words = [tar[-1] for tar in observation_['fields'] if tar[0] != 'button']

        # Extract target from fields
        dom_elements = observation_['dom_elements']

        if self.target_ele_ind == 0:
            for ele_id, element in enumerate(dom_elements):
                if element['tag'].strip() == 't':
                    self.check_boxes.append(dom_elements[ele_id-1])
                    self.check_boxes_texts.append(element['text'])
                elif element['tag'].strip() == 'button':
                    self.submit_btn = element

            self.match_matrix = np.zeros((len(prompt_words), len(self.check_boxes_texts)))

            for j, prompt_word in enumerate(prompt_words):
                for k, check_boxes_text in enumerate(self.check_boxes_texts):
                    self.match_matrix[j, k] = compare_strings(prompt_word, check_boxes_text)

        max_values = np.max(self.match_matrix, axis=1)
        max_indices = np.argmax(self.match_matrix, axis=1)

        threshold = 0.7
        matches = np.where(max_values > threshold, max_indices, -1)

        # print(matches, self.target_ele_ind)

        if self.target_ele_ind < len(matches):
            action = env.unwrapped.create_action(
                ActionTypes.CLICK_ELEMENT,
                ref=self.check_boxes[matches[self.target_ele_ind]]["ref"]
            )
            self.target_ele_ind += 1
        else:
            action = env.unwrapped.create_action(
                ActionTypes.CLICK_ELEMENT,
                ref=self.submit_btn["ref"]
            )
            self.target_ele_ind = 0
            self.submit_btn = None
            self.check_boxes = []
            self.check_boxes_texts = []
            self.match_matrix = None

        time.sleep(0.5)

        return action


# This is the main entry
# env = gymnasium.make('miniwob/click-test-2-v1', render_mode='human')
# env = gymnasium.make('miniwob/click-test-2-v1', render_mode='human')
# circle-center
env = gymnasium.make('miniwob/click-checkboxes-soft', render_mode='human')
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

    # print(observation["utterance"], observation["fields"])
    
    final_reward = 0

    # run for some time
    for i in range(10000):
        action = policy(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        # print(f"Step {i}: Reward={reward}, Position={action['coords']}")

        if terminated or truncated:
            print(f"Episode finished with final reward: {reward}")
            time.sleep(3)
            policy.target_pos = None
            policy.current_pos = None
            observation, info = env.reset()
            print(observation["utterance"], observation["fields"])
finally:
    env.close()