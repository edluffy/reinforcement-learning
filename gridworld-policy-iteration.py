import environments
import random

def policy_iteration(env, gamma=0.9, theta=0.0001):
    n = 0
    policy_stable = False
    value =  {s: 0 for s in env.states}
    policy = {s: 'R' for s in env.states}

    while not policy_stable:
        n+=1
        print('Iteration: ', n)
        env.display(value)
        env.display(policy)

        # Evaluate policy
        while True:
            delta = 0
            for state in env.states:
                old_value = value[state]
                # pi(a|s) = 0.25, p(s',r|s,a) = 1
                value[state] = 0.25*sum(r+gamma*value[ns] for (s, a), (ns, r) in env.model.items() if s == state)
                delta = max(delta, abs(old_value-value[state]))
            if delta < theta:
                break

        # Improve policy
        policy_stable = True
        for state in env.states:
            old_action = policy[state]
            # p(s',r|s,a) = 1
            action_values = {a: r+gamma*value[ns] for (s, a), (ns, r) in env.model.items() if s == state}
            policy[state] = max(action_values, key=action_values.get)

            if old_action != policy[state]:
                policy_stable = False
                break

env = environments.GridWorld()
policy_iteration(env)
