import environments
import random

argmax = lambda d: max(d, key=d.get)

# Every-visit Monte Carlo evaluation with incremental updates to action-value function
def monte_carlo(env, gamma=0.9, alpha=0.5, epsilon=0.1, ep=1000):
    Q = {(s, a): 0 for (s, a) in env.model.keys()}
    policy = {s: random.choice(env.actions) for s in env.states}

    for _ in range(ep):
        # Generate episode
        episode = [] # MRP: [(s0, a0, r1), (s1, a1, r2),...]
        s = random.choice(env.states)
        while not env.terminal(s):
            # epsilon greedy
            a = policy[s] if random.random() > epsilon else random.choice(env.actions)

            (ns, r) = env.model[(s, a)]
            episode.append((s, a, r))
            s = ns

        # Back-sample through episode
        G = 0
        for (s, a, r) in reversed(episode):
            G = gamma*G + r
            Q[(s, a)] += alpha*(G-Q[(s, a)])
            policy[s] = argmax({a: Q[(s, a)] for a in env.actions})

    V = {s: max(Q[(s, a)] for a in env.actions) for s in env.states}

    env.display(V)
    env.display(policy)

env = environments.GridWorld()
monte_carlo(env)


