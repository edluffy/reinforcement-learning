import environments
import random

argmax = lambda d: max(d, key=d.get)

def q_learning(env, gamma=0.9, alpha=0.5, epsilon=0.1, ep=200):
    Q = {(s, a): 0 for (s, a) in env.model.keys()}

    def epsilon_greedy(s):
        if random.random() > epsilon:
            return argmax({a: Q[(s, a)] for a in env.actions})
        else:
            return random.choice(env.actions)

    for _ in range(ep):
        s = random.choice(env.states)
        while not env.terminal(s):
            a = epsilon_greedy(s)
            (ns, r) = env.model[(s, a)]
            Q[(s, a)] += alpha*(r + gamma*max(Q[(ns, _a)] for _a in env.actions)-Q[(s, a)])
            s = ns

    # Extract optimal state-value function + optimal policy
    V = {s: max(Q[(s, a)] for a in env.actions) for s in env.states}
    policy = {s: argmax({a: Q[(s, a)] for a in env.actions}) for s in env.states}

    env.display(V)
    env.display(policy)

env = environments.GridWorld()
q_learning(env)
