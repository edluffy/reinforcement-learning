import environments

argmax = lambda d: max(d, key=d.get)

def value_iteration(env, gamma=0.9, theta=0.0001):
    V = {s: 0 for s in env.states}
    policy = {s: 0 for s in env.states}
    n = 0

    while True:
        n += 1
        delta = 0
        for s in env.states:
            old_V = V[s]
            # p(s',r|s,a) = 1
            V[s] = max(r + gamma*V[ns] for (ns, r) in [env.model[(s, a)] for a in env.actions])
            delta = max(delta, abs(old_V-V[s]))
        if delta < theta:
            break

    # Find optimal policy
    for state in env.states:
        policy[state] = argmax({a: r + gamma*V[ns] for (s, a), (ns, r) in env.model.items() if s == state})

    print('Iterations: ', n)
    env.display(V)
    env.display(policy)

    return policy

env = environments.GridWorld()
value_iteration(env)
