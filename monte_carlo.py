import environments
import random

argmax = lambda d: max(d, key=d.get)

# Every-visit Monte Carlo policy evaluation with incremental updates to value function
def monte_carlo(env, policy, gamma=0.9, alpha=None, ep=500):
    value =  {s: 0 for s in env.states}
    counter = {s: 0 for s in env.states}
    returns = {s: [] for s in env.states}

    for i in range(ep):
        # Generate episode
        episode = [] # MRP: [(s0, a0, r1), (s1, a1, r2),...]
        s = random.choice(env.states)
        while True:
            a = policy[s]
            (ns, r) = env.model[(s, a)]
            episode.append((s, a, r))

            if env.terminal(s): break
            s = ns

        # Back-sample through episode
        ret = 0
        for (state, action, reward) in reversed(episode):
            # Check if first occurence
            ret = gamma*ret + reward
            if alpha:
                value[state] += alpha*(ret-value[state])
            else:
                counter[state] = counter[state]+1
                value[state] += (1/counter[state])*(ret-value[state])

    env.display(value)

# Evaluation of optimal policy
policy = {(0, 0): 'U', (1, 0): 'L', (2, 0): 'L', (3, 0): 'D',
          (0, 1): 'U', (1, 1): 'U', (2, 1): 'U', (3, 1): 'D',
          (0, 2): 'U', (1, 2): 'U', (2, 2): 'D', (3, 2): 'D',
          (0, 3): 'U', (1, 3): 'R', (2, 3): 'R', (3, 3): 'D'}

env = environments.GridWorld()
monte_carlo(env, policy, gamma=0.9)


