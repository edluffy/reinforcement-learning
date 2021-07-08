import gym
import numpy as np

argmax = lambda d: max(d, key=d.get)

def vfa_monte_carlo(env, gamma=0.9, alpha=0.1, ep=10):
    obs = env.reset()
    w = np.random.random(5)

    # Generate episode
    episode = [] # [(x0, a0, r1), (x1, a1, r2),...]
    for n in range(ep):
        while True:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            episode.append((obs, action, reward))

            if done:
                obs = env.reset()
                break

        # Q_approx: action-value function approximation for current state and action
        #        x: feature vector for current state and action
        G = 0
        for (obs, action, reward) in reversed(episode):
            x = np.append(obs, action)
            G = gamma*G + reward
            Q_approx = np.dot(x, w)
            w += alpha*(G-Q_approx)*x
        #print('MSE:', (G-Q_approx)**2)
    print(w)

    for n in range(1):
        for t in range(100000):
            env.render()
            action = argmax({a: np.dot(np.append(obs, a), w) for a in [0, 1]})

            obs, reward, done, _ = env.step(action)

            if done:
                print('Episode', n, 't:', t)
                obs = env.reset()
                break

    env.close()
    


env = gym.make("CartPole-v1")
for _ in range(10):
    vfa_monte_carlo(env, ep=200)

