import gym
import numpy as np
import random

argmax = lambda d: max(d, key=d.get)

def semi_gradient_sarsa(env, gamma=0.9, alpha=0.1, epsilon=0.1, ep=30000):
    obs = env.reset()
    weights = np.random.random((2, 4))

    alpha = np.linspace(0.4, 0.01, ep)
    #epsilon = np.linspace(1, 0.1, ep)

    def epsilon_greedy():
        if random.random() > epsilon:
            return argmax({a: obs @ weights[a] for a in [0, 1]})
        else:
            return env.action_space.sample()

    for n in range(ep):
        action = epsilon_greedy()

        t = 0
        while True:
            #env.render()

            #obs, reward, done, _ = env.step(action)
            features = obs
            q_hat = features @ weights[action]

            action_next = epsilon_greedy()

            obs_next, reward, done, _ = env.step(action_next)
            features_next = obs_next
            q_hat_next = features_next @ weights[action_next]

            weights[action] += alpha[n]*(reward + gamma*q_hat_next - q_hat)*features

            obs = obs_next
            action = action_next

            t += 1

            if done:
                print(t)
                obs = env.reset()
                break

    env.close()

env = gym.make("CartPole-v1")
semi_gradient_sarsa(env)
