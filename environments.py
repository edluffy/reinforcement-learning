class GridWorld():
    def __init__(self):
        #  +---+---+---+---+
        #  | X |   |   |   |
        #  +---+---+---+---+
        #  |   |   |   |   |
        #  +---+---+---+---+
        #  |   |   |   | X |
        #  +---+---+---+---+
        #  X: terminal state

        # 'S': set of self.states with Markov property
        self.states = ((0, 0), (1, 0), (2, 0), (3, 0),
                       (0, 1), (1, 1), (2, 1), (3, 1),
                       (0, 2), (1, 2), (2, 2), (3, 2),
                       (0, 3), (1, 3), (2, 3), (3, 3))

        # 'A': set of self.actions agent can take
        self.actions = ('U', 'D', 'L', 'R')

        # 'R': gives immediate expected reward (in next state) from state s
        self.rewards = {s: -1 for s in self.states}
        self.rewards[(0, 0)] = 0
        self.rewards[(3, 3)] = 0

        # 'P': state transition probability p(s',r|s,a)
        self.prob = 0.1

        # Deterministic model of the environment
        delta = {'U': (0, -1), 'D': (0,  1), 'L': (-1, 0), 'R': (1,  0)}
        self.model = {} # {(state, action): (next_state, reward)}
        for s in self.states:
            for a in self.actions:
                ns = (s[0]+delta[a][0], s[1]+delta[a][1])
                ns = (max(ns[0], 0), max(ns[1], 0))
                ns = (min(ns[0], 3), min(ns[1], 3))
                self.model[(s, a)] = (ns, self.rewards[s])

    def terminal(self, state): # Check if in a terminal state
        return state == (0, 0) or state == (3, 3)

    def display(self, var): # Display a dictionary of values with states as keys
        _type = [type(k) for k in var.values()][0]
        if _type == float or _type == int:
            fstr = '%6.2f'
        else:
            fstr = '%6c'

        for i, s in enumerate(self.states):
            if i % 4 == 0:
                print('\n'+'-'*35)
            print(fstr % var[s], end=' | ')
        print('\n'+'-'*35)
        print('\n\n\n')
