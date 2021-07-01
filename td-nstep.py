import environments
import random

def td_nstep(env, n=5, gamma=0.9, alpha=None):
    value = {s: 0 for s in env.states}
    counter = {s: 0 for s in env.states}

    for i in range(1):
        # Generate episode
        t = 0
        T = float('inf')
        s = random.choice(env.states)
        episode = [] # MRP: [(s0, a0, r1), (s1, a1, r2),...]
        while not env.terminal(s):
            # Take action, get next state, rewards
            a = random.choice(env.actions)
            (ns, r) = env.model[(s, a)]
            episode.append((s, a, r))
            if env.terminal(ns): T = t+1
            s = ns

            # Update value estimate for state at time tau
            tau = t-n
            if tau >= 0:
                # sum disounted rewards from time tau+1 -> tau+n (present time t), or episode end if smaller
                print(min(tau+n,T))
                td_target = sum(gamma**i * r for i, (_, _, r) in enumerate(episode[tau:min(tau+n, T)]))\
                        + gamma**n * value[episode[min(tau+n, T)][0]]

                (state, _, _) = episode[tau]
                td_error = td_target - value[state]
                counter[state] += 1
                value[state] += (1/counter[state])*(td_error)
            t+=1

    env.display(value)


env = environments.GridWorld()
td_nstep(env)
