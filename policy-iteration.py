import environments
import random

argmax = lambda d: max(d, key=d.get)

def policy_iteration(env, gamma=0.9, theta=0.0001):
    value =  {s: 0 for s in env.states}
    policy = {s: random.choice(env.actions) for s in env.states}
    policy_stable = False
    n = 0

    while not policy_stable:
        n+=1
        # Evaluate policy
        while True:
            delta = 0
            for state in env.states:
                old_value = value[state]
                # p(s',r|s,a) = 1
                (next_state, reward) = env.model[state, policy[state]]
                value[state] = reward+gamma*value[next_state]
                delta = max(delta, abs(old_value-value[state]))
            if delta < theta:
                break

        # Improve policy
        for state in env.states:
            old_action = policy[state]
            policy[state] = argmax({a: r+gamma*value[ns] for (s, a), (ns, r) in env.model.items() if s == state})

            policy_stable = (old_action == policy[state])
            if not policy_stable:
                break

    print('Iterations: ', n)
    env.display(value)
    env.display(policy)

env = environments.GridWorld()
policy_iteration(env)
