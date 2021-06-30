import environments

def value_iteration(env, gamma=0.9, theta=0.0001):
    argmax = lambda d: max(d, key=d.get)
    value = {s: 0 for s in env.states}
    policy = {s: 0 for s in env.states}
    n = 0

    while True:
        n += 1
        delta = 0
        for state in env.states:
            old_value = value[state]
            # pi(a|s) = 0.25, p(s',r|s,a) = 1
            value[state] = 0.25*sum(r + gamma*value[ns] for (s, _), (ns, r) in env.model.items() if s == state)
            delta = max(delta, abs(old_value-value[state]))
        if delta < theta:
            break

    # Find optimal policy
    for state in env.states:
        policy[state] = argmax({a: r + gamma*value[ns] for (s, a), (ns, r) in env.model.items() if s == state})

    print('Iterations: ', n)
    env.display(value)
    env.display(policy)

env = environments.GridWorld()
value_iteration(env)
