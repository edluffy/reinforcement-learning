import random
#  +---+---+---+---+
#  | X |   |   |   |
#  +---+---+---+---+
#  |   |   |   |   |
#  +---+---+---+---+
#  |   |   |   | X |
#  +---+---+---+---+
#  X: terminal state

# A Markov Decision Process: <S, A, P, R, Gamma>

# 'S': set of states with Markov property
states = ((0, 0), (1, 0), (2, 0), (3, 0),
          (0, 1), (1, 1), (2, 1), (3, 1),
          (0, 2), (1, 2), (2, 2), (3, 2),
          (0, 3), (1, 3), (2, 3), (3, 3))

# 'A': set of actions agent can take
actions = ('U', 'D', 'L', 'R')

# 'P': state transition probability matrix
transitions = None # Leave blank if fully deterministic

# 'R': gives immediate expected reward (in next state) from state s
rewards = {s: -1 for s in states}
rewards[(0, 0)] = 0
rewards[(3, 3)] = 0

# 'Gamma': the discount factor
gamma = 0.9

env = {} # {(state, action): (next_state, reward)}

for s in states:
    for a in actions:
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

        env[(s, a)] = (ns, rewards[s])

value =  {s: 0 for s in states}
policy = {s: 'R' for s in states}
counter = {s: 0 for s in states}

# Every-visit Monte Carlo with incremental updates to value function
for i in range(50000):
    # Generate episode
    episode = [] # MRP: [(s0, a0, r1), (s1, a1, r2),...]
    s = random.choice(states)
    while True:
        a = random.choice(actions)
        (ns, r) = env[(s, a)]
        episode.append((s, a, r))
        s = ns
        if r == 0: break

    # Back-sample through episode
    ret = 0
    for (state, action, reward) in reversed(episode):
        ret = gamma*ret + reward
        counter[state] = counter[state]+1
        value[state] += (1/counter[state])*(ret-value[state])

def print_grid(var):
    if var == policy:
        fstr = '%6c'
    else:
        fstr = '%6.2f'

    for i, s in enumerate(states):
        if i % 4 == 0:
            print('\n'+'-'*35)
        print(fstr % var[s], end=' | ')
    print('\n'+'-'*35)
    print('\n\n\n')

print_grid(value)
