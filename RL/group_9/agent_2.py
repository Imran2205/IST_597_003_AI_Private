import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import word_tokenize
import gymnasium
import miniwob
from gymnasium import spaces
import time
from miniwob.action import ActionTypes


def load_pretrained_model():
    """Load pre-trained Word2Vec model from HuggingFace"""
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens', model_max_length=512)
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    return model, tokenizer

def get_sentence_embedding(text, model, tokenizer):
    """Generate embedding for text using the pre-trained model"""
    # Tokenize and encode
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    # Use mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()


def compare_strings(str1, str2, model, tokenizer):
    """Compare two strings using pre-trained embeddings"""
    # Load model and tokenizer
    model, tokenizer = load_pretrained_model()
    # Get embeddings
    emb1 = get_sentence_embedding(str1, model, tokenizer)
    emb2 = get_sentence_embedding(str2, model, tokenizer)
    # Calculate similarity
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity

def parse_utterance(utterance):
    lower_utter = utterance.lower()
    if "select nothing" in lower_utter:
        return []
    if "select " in lower_utter and "and click submit" in lower_utter:
        start = lower_utter.index("select ") + len("select ")
        end = lower_utter.index("and click submit")
        targets_str = utterance[start:end].strip()
        targets = [t.strip() for t in targets_str.split(",")]
        return targets
    else:
        return []

def find_checkboxes_and_labels(dom_elements):
    checkboxes = []
    for element in dom_elements:
        if element.get('tag', '') == 'input' and 'checkbox' in element.get('attributes', {}).get('type', ''):
            checkbox_text = element.get('text', '').strip()
            x = element['left'][0] + element['width'][0] / 2
            y = element['top'][0] + element['height'][0] / 2
            checkboxes.append((checkbox_text, x, y))
    return checkboxes

def find_submit_button(dom_elements):
    for element in dom_elements:
        text = element.get('text', '').strip().lower()
        if 'submit' in text:
            x = element['left'][0] + element['width'][0] / 2
            y = element['top'][0] + element['height'][0] / 2
            return (x, y)
    return None

def choose_checkboxes(target_texts, checkboxes, model, tokenizer):
    chosen = []
    for target in target_texts:
        best_score = -1
        best_checkbox = None
        for (cb_text, x, y) in checkboxes:
            if cb_text:
                print(cb_text, checkboxes)
                sim = compare_strings(target, cb_text, model, tokenizer)
                if sim > best_score:
                    best_score = sim
                    best_checkbox = (cb_text, x, y)
        if best_checkbox:
            chosen.append(best_checkbox)
    return chosen

class TimeBasedRewardWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_distance = None
        self.start_time = None
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

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.start_time = time.time()
        target = dict(observation['fields']).get('target', '')
        if target:
            target_pos = self.get_target_position(
                observation['dom_elements'],
                target
            )
            self.previous_distance = self.calculate_distance(
                np.array([0, 0]),
                target_pos
            )
        else:
            self.previous_distance = None

        self.step_rewards = []
        self.final_reward = 0.0
        return observation, info

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        current_pos = action['coords']

        target = dict(observation['fields']).get('target', '')
        if target:
            target_pos = self.get_target_position(
                observation['dom_elements'],
                target
            )
        else:
            target_pos = None

        if target_pos is not None:
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
        else:
            # If no single target is found, fallback to no distance-based reward
            reward = 0.0

        for element in observation['dom_elements']:
            if element['text'].strip() == target.strip():
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


class GradualMovePolicy:
    def __init__(self, env, step_size=5):
        self.env = env
        self.step_size = step_size
        self.target_pos = None
        self.origin = np.array([0, 0])
        self.current_pos = None

        # Additional attributes for checkbox scenario
        self.model, self.tokenizer = load_pretrained_model()
        self.stage = "parse_targets"  # parse_targets -> click_checkboxes -> click_submit
        self.target_texts = []
        self.chosen_checkboxes = []
        self.current_checkbox_index = 0
        self.submit_coords = None

    def initialize_position(self, observation):
        if self.current_pos is None:
            self.current_pos = self.origin

    def move_toward_point(self, point):
        window_height = self.env.observation_space["screenshot"].shape[0]
        window_width = self.env.observation_space["screenshot"].shape[1]

        current = np.array(self.current_pos)
        target = np.array(point)

        direction = target - current
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            # Close enough to click
            action = {
                "action_type": 1,  # CLICK_COORDS
                "coords": self.current_pos,
                "key": 0
            }
            return action, True
        else:
            if distance > 0:
                direction = direction / distance * self.step_size
            noise = np.random.normal(0, self.step_size * 0.5, size=2)
            new_pos = current + direction + noise
            self.current_pos = np.clip(new_pos, [0, 0], [window_width, window_height])

            # Optional visualization
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
            self.env.unwrapped.instance.driver.execute_script(exec_js)

            action = {
                "action_type": 2,  # MOVE_COORDS
                "coords": self.current_pos,
                "key": 0
            }
            return action, False

    def __call__(self, observation):
        self.initialize_position(observation)
        dom_elements = observation['dom_elements']
        utterance = observation["utterance"]

        if self.stage == "parse_targets":
            # Parse targets from utterance
            self.target_texts = parse_utterance(utterance)
            checkboxes = find_checkboxes_and_labels(dom_elements)
            self.chosen_checkboxes = choose_checkboxes(self.target_texts, checkboxes, self.model, self.tokenizer)
            self.submit_coords = find_submit_button(dom_elements)
            # If no targets, skip directly to submit stage
            if len(self.target_texts) == 0:
                self.stage = "click_submit"
            else:
                self.stage = "click_checkboxes"
            # Return a no-op action to proceed
            return {
                "action_type": 2,
                "coords": self.current_pos,
                "key": 0
            }

        elif self.stage == "click_checkboxes":
            if self.current_checkbox_index < len(self.chosen_checkboxes):
                # Move towards next checkbox
                _, x, y = self.chosen_checkboxes[self.current_checkbox_index]
                action, clicked = self.move_toward_point([x, y])
                if clicked:
                    # Once clicked, go to next checkbox
                    self.current_checkbox_index += 1
                    # If done with all, move to submit
                    if self.current_checkbox_index >= len(self.chosen_checkboxes):
                        self.stage = "click_submit"
                return action
            else:
                # If no checkboxes or done with them, move to submit stage
                self.stage = "click_submit"
                return {
                    "action_type": 2,
                    "coords": self.current_pos,
                    "key": 0
                }

        elif self.stage == "click_submit":
            if self.submit_coords is not None:
                action, clicked = self.move_toward_point(self.submit_coords)
                return action
            else:
                # No submit found, just do nothing
                return {
                    "action_type": 2,
                    "coords": self.current_pos,
                    "key": 0
                }


env = gymnasium.make('miniwob/click-checkboxes-v1', render_mode='human')
env.unwrapped.instance = env.unwrapped._hard_reset_instance()
env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=99999999999)

env = TimeBasedRewardWrapper(env)

try:
    policy = GradualMovePolicy(env, step_size=5)
    observation, info = env.reset(seed=42)

    print(observation["utterance"], observation["fields"])

    for i in range(10000):
        action = policy(observation)
        time.sleep(0.2)
        observation, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            print(f"Episode finished with final reward: {reward}")
            time.sleep(3)
            # Reset for next episode
            policy.target_pos = None
            policy.current_pos = None
            policy.stage = "parse_targets"
            policy.current_checkbox_index = 0
            observation, info = env.reset()
            print(observation["utterance"], observation["fields"])
finally:
    env.close()
