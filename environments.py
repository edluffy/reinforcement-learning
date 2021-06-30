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

        # Model of the environment
        self.model = {} # {(state, action): (next_state, reward)}
        for s in self.states:
            for a in self.actions:
                if a == 'U':
                    ns = s if s[1] == 0 else (s[0], s[1]-1)
                elif a == 'D':
                    ns = s if s[1] == 3 else (s[0], s[1]+1)
                elif a == 'L':
                    ns = s if s[0] == 0 else (s[0]-1, s[1])
                elif a == 'R':
                    ns = s if s[0] == 3 else (s[0]+1, s[1])

                if s == (0, 0) or s == (3, 3):
                    ns = s

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
