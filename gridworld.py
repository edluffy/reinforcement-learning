import numpy as np

# A Markov Decision Process: <S, A, P, R, Gamma>
class GridWorld():
    def __init__(self):
        self.states = [[0, 0], [1, 0], [2, 0], [3, 0],
                       [0, 1], [1, 1], [2, 1], [3, 1],
                       [0, 2], [1, 2], [2, 2], [3, 2]]

        self.actions = ['U','D','L','R']

        self.transition_pr = None # Leave blank if fully deterministic

        self.rewards = [0, 0, 0,  1,
                        0, 0, 0, -1,
                        0, 0, 0,  0]

        self.gamma = 0.99

        self.cstate = self.states[0]

    def sample(self, action):
        if action not in self.actions:
            print('Unknown action')

        nstate = self.cstate

        if action == 'U' and nstate[1] > 0:
            nstate[1] = nstate[1]-1
        elif action == 'D' and nstate[1] < 2:
            nstate[1] = nstate[1]+1
        elif action == 'L' and nstate[0] > 0:
            nstate[0] = nstate[0]-1
        elif action == 'R' and nstate[0] < 3:
            nstate[0] = nstate[0]+1

        obs = nstate
        reward = self.rewards[self.states.index(nstate)]
        done = (nstate == [3, 0])

        return obs, reward, done

env = GridWorld()

print(env.sample('U'))
