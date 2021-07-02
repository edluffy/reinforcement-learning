import environments
import random

def td_nstep(env, policy, n=0, gamma=0.9, alpha=None, ep=100):
    value = {s: 0 for s in env.states}
    counter = {s: 0 for s in env.states}

    for i in range(ep):
        # Generate episode
        t = 0
        T = float('inf')
        s = random.choice(env.states)
        episode = [] # MRP: [(s0, a0, r1), (s1, a1, r2),...]
        while True:
            # Take action, get next state, rewards
            if t < T:
                a = policy[s]
                (ns, r) = env.model[(s, a)]
                episode.append((s, a, r))
                if env.terminal(ns):
                    T = t+1
                s = ns

            # Update value estimate for state at time tau
            tau = t-n-1
            if tau >= 0:
                td_target = 0
                # sum discounted rewards from time tau -> tau+n (present time t), or episode end if smaller
                for i, (state, _, reward) in enumerate(episode[tau:min(tau+n+1, T)]):
                    td_target += gamma**i * reward

                if tau+n+1 < T:
                    td_target += gamma**(n+1) * value[episode[tau+n+1][0]]

                state = episode[tau][0]
                counter[state] += 1
                td_error = td_target - value[state]
                value[state] += (1/counter[state])*(td_error)

            if tau == T-1: break
            t+=1

    env.display(value)

# Evaluation of optimal policy
policy = {(0, 0): 'U', (1, 0): 'L', (2, 0): 'L', (3, 0): 'D',
          (0, 1): 'U', (1, 1): 'U', (2, 1): 'U', (3, 1): 'D',
          (0, 2): 'U', (1, 2): 'U', (2, 2): 'D', (3, 2): 'D',
          (0, 3): 'U', (1, 3): 'R', (2, 3): 'R', (3, 3): 'D'}

env = environments.GridWorld()
td_nstep(env, policy)
