import tensorflow as tf
import numpy as np
import gym

class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh):
        self.w = tf.Variable(tf.random.normal(shape=(M1,M2)))
        self.b = tf.Variable(tf.zeros(M2, dtype=tf.float32))
        self.f = f

    def forward(self, x):
        return self.f(x @ self.w + self.b)
    

class DQN:
    def __init__(self, D, K, layer_sizes):
        self.layers = [HiddenLayer(M1, M2) for M1, M2 in zip([D]+layer_sizes, layer_sizes+[K])]
        self.layers[-1].f = lambda x: x

        self.x = tf.Variable(tf.zeros((1, D)))

        z = self.x
        for layer in self.layers:
            z = layer.forward(z)
        y_hat = z

        self.experience = [] # [(s0, a0, r1, s1, done), (s1, a1, r2, s2, done),...]

    def train(self):
        batch = self.experience[-5:]
        targets = [r + 0.9*]

    def play(self, env, gamma=0.9, alpha=0.5, epsilon=0.1):
        state = env.reset()

        done = False
        while not done:
            if np.random.random() > 0.5:
                action = env.action_space.sample()
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            self.experience.append((state, action, reward, next_state, done))

            state = next_state

        self.train()

        #selected_action_values = tf.reduce_sum(y_hat * tf.one_hot(self.actions, K), reduction_indices=[1])
        #cost = tf.reduce_sum(tf.square(self.G-selected_action_values))
        

tf.compat.v1.enable_eager_execution()
env = gym.make("CartPole-v1")

D = len(env.observation_space.sample())
K = env.action_space.n
agent = DQN(D, K, [200, 200])
agent.play(env)
