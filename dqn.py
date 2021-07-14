from collections import deque
import tensorflow as tf
import numpy as np
import random
import gym

# Deep Q-Learning with Experience Replay
class DQN:
    def __init__(self, D, K, layer_sizes, replay_size, batch_size):

        # Model
        x_in = tf.keras.Input([D,])
        x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x_in)
        x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Dense(K, activation='linear', kernel_initializer='he_uniform')(x)

        self.model = tf.keras.Model(inputs=x_in, outputs=x, name='Deep Q-Learning with Experience Replay')
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        self.model.summary()

        # Replay Memory [(s0, a0, r1, s1, done), (s1, a1, r2, s2, done),...]
        self.replay_memory = deque(maxlen=replay_size) 
        self.batch_size = batch_size

    def replay(self):
        batch = random.sample(self.replay_memory, min(self.batch_size, len(self.replay_memory)))
        targets = []
        states = []

        batch = {}
        for k in ['state', 'action', 'reward', 'next_state', 'done']:
            batch[k] = random.sample(self.replay_memory[k], min(self.batch_size, len(self.replay_memory)))
        print(batch['state'])

        for (state, action, reward, next_state, done) in batch:
            if done:
                target = reward
            else:
                target = reward + 0.9*tf.reduce_max(self.model.predict(next_state))

            targets.append(target)
            states.append(state)

        cost = lambda: tf.square(target-tf.reduce_max(self.model.predict(state)))

        target = tf.Variable(targets)
        state = tf.Variable(states)
        self.model.fit(state, target, batch_size=self.batch_size)

    def policy(self, state, epsilon):
        if random.random() > epsilon:
            return env.action_space.sample()
        else:
            return tf.math.argmax(self.model.predict(state), 1).numpy()[0]


    def run(self, env, gamma=0.9, alpha=0.5, epsilon=0.1):
        state = env.reset()
        state = np.reshape(state, [1, len(state)])

        done = False
        t = 0
        while not done:
            t += 1
            action = self.policy(state, epsilon)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, len(next_state)])

            self.replay_memory.append((state, action, reward, next_state, done))

            state = next_state

        print(t)
        self.replay()

        #selected_action_values = tf.reduce_sum(y_hat * tf.one_hot(self.actions, K), reduction_indices=[1])
        #cost = tf.reduce_sum(tf.square(self.G-selected_action_values))
        

tf.compat.v1.enable_eager_execution()
env = gym.make("CartPole-v1")

D = len(env.observation_space.sample())
K = env.action_space.n
agent = DQN(D, K, layer_sizes=[200, 200], replay_size=2000, batch_size=32)
for _ in range(3):
    agent.run(env)
