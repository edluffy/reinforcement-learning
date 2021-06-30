import environments
import random

# Every-visit Monte Carlo with incremental updates to value function
def monte_carlo(env, gamma=0.9, alpha=None, ep=50000):
    value =  {s: 0 for s in env.states}
    counter = {s: 0 for s in env.states}

    for i in range(ep):
        # Generate episode
        episode = [] # MRP: [(s0, a0, r1), (s1, a1, r2),...]
        s = random.choice(env.states)
        while not env.terminal(s):
            a = random.choice(env.actions)
            (ns, r) = env.model[(s, a)]
            episode.append((s, a, r))
            s = ns

        # Back-sample through episode
        ret = 0
        for (state, action, reward) in reversed(episode):
            ret = gamma*ret + reward
            if alpha:
                value[state] += alpha*(ret-value[state])
            else:
                counter[state] = counter[state]+1
                value[state] += (1/counter[state])*(ret-value[state])

    env.display(value)

env = environments.GridWorld()
monte_carlo(env, gamma=0.9)

