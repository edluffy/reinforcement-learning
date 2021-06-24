import numpy as np

# A Markov Decision Process: <S, A, P, R, Gamma>
class GridWorld():
    def __init__(self):
        self.states = np.array([[0, 0], [0, 1], [0, 2], [0, 3],
                                [1, 0], [1, 1], [1, 2], [1, 3],
                                [2, 0], [2, 1], [2, 2], [2, 3]])

        self.actions = np.array(['U','D','L','R'])

        self.transition_pr = None # Leave blank if fully deterministic

        self.rewards = np.array([0, 0, 0,  1,
                                 0, 0, 0, -1,
                                 0, 0, 0,  0])
        self.gamma = 0.99

        self.current_state = self.states[0]

    def sample(self, action):
        if action in self.actions:
            # wrap around if action causes border collision
            if (self.current_state[0] == 0 and action == 'U') \
                    or (self.current_state[0] == 1 and action == 'D') \
                    or (self.current_state[1] == 0 and action == 'L') \
                    or (self.current_state[1] == 3 and action == 'R'):
                next_state = self.current_state
        ##return obs, reward, done


env = GridWorld()

env.sample('U')
