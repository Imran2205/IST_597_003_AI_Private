import miniwob.action as actions

# Print action type integer values
action_types = actions.ActionTypes
action_map = {}

for action in dir(actions):
    if action.isupper():  # Get only constants
        value = getattr(actions, action)
        if isinstance(value, int):
            action_map[action] = value

print("Action type integer values:", action_map)