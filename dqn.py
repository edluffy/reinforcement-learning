from collections import deque
import tensorflow as tf
import numpy as np
import random
import gym

# Deep Q-Learning with Experience Replay
class DQN:
    def __init__(self, input_size, output_size, replay_size=2000, batch_size=32,
            gamma=0.99, alpha=0.01, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.01):

        self.input_size = input_size
        self.output_size = output_size

        # Hyperparams
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Model
        x_in = tf.keras.Input([self.input_size,])
        x = tf.keras.layers.Dense(24, activation='relu')(x_in)
        x = tf.keras.layers.Dense(24, activation='relu')(x)
        x = tf.keras.layers.Dense(self.output_size, activation='linear')(x)
        self.model = tf.keras.Model(inputs=x_in, outputs=x, name='DQN')
        self.optimizer = tf.keras.optimizers.Adam(self.alpha)
        self.model.summary()

        # Replay Memory [(s0, a0, r1, s1, done), (s1, a1, r2, s2, done)...]
        self.replay_memory = deque(maxlen=replay_size)
        self.batch_size = batch_size


    def policy(self, state):
        if random.random() > self.epsilon:
            return np.argmax(self.model.predict(state))
        else:
            return env.action_space.sample()
        
    def run(self, env, episodes, log_dir):
        log_writer = tf.summary.create_file_writer(log_dir)
        episode_rewards = deque(maxlen=100)
        episode_losses = deque(maxlen=100)

        for ep in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, len(state)])

            # Run Episode
            done = False
            t = 0
            while not done:
                t += 1
                #env.render()

                action = self.policy(state)

                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, len(next_state)])

                if done:
                    reward = -10

                self.replay_memory.append((state, action, reward, next_state, done))
                loss = self.replay()

                episode_rewards.append(reward)
                episode_losses.append(loss)

                state = next_state
            print('episode:', ep, 'score:', t, 'e:', self.epsilon)

            # Logging
            with log_writer.as_default():
                tf.summary.scalar('mean (last 100) episode rewards', np.mean(episode_rewards), step=ep)
                tf.summary.scalar('mean (last 100) episode losses', np.mean(episode_losses), step=ep)
                tf.summary.scalar('episode score', np.mean(t), step=ep)
                tf.summary.scalar('epsilon', self.epsilon, step=ep)

            # Adjust epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def replay(self):
            batch = random.sample(self.replay_memory, min(self.batch_size, len(self.replay_memory)))
            states, actions, rewards, next_states, dones = zip(*batch)

            with tf.GradientTape() as tape:
                Qs_actual = rewards + self.gamma*np.max(self.model(np.vstack(next_states)))*(not dones)
                Qs_selected = tf.reduce_sum(self.model(np.vstack(states))*tf.one_hot(actions, self.output_size), axis=1)
                loss = tf.math.reduce_mean(tf.square(Qs_actual-Qs_selected))

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            return loss


tf.compat.v1.enable_eager_execution()
env = gym.make("CartPole-v1")

o_space = len(env.observation_space.sample())
a_space = env.action_space.n

agent = DQN(input_size=o_space, output_size=a_space, epsilon_decay=0.995, alpha=0.01)
agent.run(env, 5000, 'logs/dqn/e=0.9999-a=0.01')
