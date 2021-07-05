import environments
import random

argmax = lambda d: max(d, key=d.get)

def sarsa(env, gamma=0.9, alpha=0.5, epsilon=0.1, ep=1000):
    action_value = {(s, a): 0 for (s, a) in env.model.keys()}

    def epsilon_greedy(state):
        if random.random() > epsilon:
            action = argmax({a: v for (s, a), v in action_value.items() if s == state})
        else:
            action = random.choice(env.actions)
        return action

    for _ in range(ep):
        state = random.choice(env.states)
        action = epsilon_greedy(state)

        while not env.terminal(state):
            (next_state, reward) = env.model[(state, action)]
            next_action = epsilon_greedy(next_state)

            td_target = reward + gamma*action_value[(next_state, next_action)]
            td_error = td_target - action_value[(state, action)]
            action_value[(state, action)] += alpha*td_error

            state = next_state
            action = next_action

    # Extract optimal state-value function + optimal policy
    policy = {}
    value = {}
    for state in env.states:
        policy[state] = argmax({a: v for (s, a), v in action_value.items() if s == state})
        value[state] = max(v for (s, _), v in action_value.items() if s == state)
    env.display(value)
    env.display(policy)

env = environments.GridWorld()
sarsa(env)
