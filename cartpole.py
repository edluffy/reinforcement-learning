import math
import gym
import numpy as np
from matplotlib import pyplot as plt

def reinforce(env, gamma=0.9, alpha_min=0.1, epsilon_min=0.1, ep=20000):
    theta = np.random.random((2, 4))
    mse = np.zeros(ep)
    times = np.zeros(ep)

    os = env.observation_space
    buckets = (1, 1, 6, 12)
    bounds = [np.linspace(os.low[n], os.high[n], buckets[n]) for n in range(4)]

    #bounds[2] = [np.linspace(np.radians(-12), np.radians(12), buckets[2])]

    alpha = np.logspace(np.log10(1), np.log10(alpha_min), ep)
    epsilon = np.logspace(np.log10(1), np.log10(epsilon_min), ep)

    Q = np.zeros(buckets + (env.action_space.n, ))
    policy = np.zeros(buckets, dtype=int)

    def discretize(obs):
        return tuple(np.argmin(np.abs(o-bounds[n])) for n, o in enumerate(obs))

    obs = env.reset()
    state = discretize(obs)

    for n in range(ep):
        # Generate episode
        episode = [] # MRP: [(s0, a0, r1), (s1, a1, r2),...]
        for t in range(99999):
            #env.render()

            # epsilon greedy
            if np.random.random() > epsilon[n]:
                action = policy[state]
            else:
                action = env.action_space.sample()

            obs, reward, done, _ = env.step(action)
            state = discretize(obs)
            episode.append((state, action, reward))

            if done:
                #print(t)
                times[n] = t
                obs = env.reset()
                break

        # Back-sample through episode
        G = 0
        for (state, action, reward) in reversed(episode):
            G = gamma*G + reward
            Q[state][action] += alpha[n]*(G-Q[state][action])
            policy[state] = np.argmax(Q[state])
        mse[n] = (G-Q[state][action])**2

    plt.plot(times)
    plt.show()

    env.close()

env = gym.make("CartPole-v1")
reinforce(env, alpha_min=0.1, epsilon_min=0.1)

#for _ in range(9999):
#    reinforce(env, alpha_min=np.random.random(), epsilon_min=np.random.random())
