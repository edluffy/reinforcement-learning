import environments
import random

argmax = lambda d: max(d, key=d.get)

def sarsa(env, gamma=0.9, alpha=0.5, epsilon=0.1, ep=200):
    Q = {(s, a): 0 for (s, a) in env.model.keys()}

    def epsilon_greedy(s):
        if random.random() > epsilon:
            return argmax({a: Q[(s, a)] for a in env.actions})
        else:
            return random.choice(env.actions)

    for _ in range(ep):
        s = random.choice(env.states)
        a = epsilon_greedy(s)
        while not env.terminal(s):
            (ns, r) = env.model[(s, a)]
            na = epsilon_greedy(ns)
            Q[(s, a)] += alpha*(r + gamma*Q[(ns, na)] - Q[(s, a)])
            s = ns; a = na

    # Extract optimal state-value function + optimal policy
    V = {s: max(Q[(s, a)] for a in env.actions) for s in env.states}
    policy = {s: argmax({a: Q[(s, a)] for a in env.actions}) for s in env.states}

    env.display(V)
    env.display(policy)

env = environments.GridWorld()
sarsa(env)
