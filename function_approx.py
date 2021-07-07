import gym
import numpy as np

argmax = lambda d: max(d, key=d.get)

def vfa_monte_carlo(env, gamma=0.9, alpha=0.1, ep=10):
    obs = env.reset()
    w = np.random.random(4)

    # Generate episode
    episode = [] # [(x0, a0, r1), (x1, a1, r2),...]
    for n in range(ep):
        for t in range(100000):
            env.render()

            action = 0 if np.dot(obs, w) < 0 else 1

            #print(action)

            obs, reward, done, _ = env.step(action)
            episode.append((obs, action, reward))

            if done:
                print('STEPS:', t)
                obs = env.reset()
                break

        # Q_approx: action-value function approximation for current state and action
        #        x: feature vector for current state and action
        G = 0
        for (obs, action, reward) in reversed(episode):
            x = obs
            G = gamma*G + reward
            Q_approx = np.dot(x, w)
            w += alpha*(G-Q_approx)*x
        print('MSE:', (G-Q_approx)**2)

    env.close()


env = gym.make("CartPole-v1")
vfa_monte_carlo(env, ep=2000)

