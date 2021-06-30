import numpy as np

# A Markov Decision Process: <S, A, P, R, Gamma>

# 'S': set of states with Markov property
states = ((0, 0), (1, 0), (2, 0), (3, 0),
          (0, 1), (1, 1), (2, 1), (3, 1),
          (0, 2), (1, 2), (2, 2), (3, 2))

# 'A': set of actions agent can take
actions = ('U', 'D', 'L', 'R')

# 'P': state transition probability matrix
transitions = None # Leave blank if fully deterministic

# 'R': gives immediate expected reward from state s
rewards = {}
for s in states:
    if s == (3, 0):
        rewards[s] = 1
    elif s == (3, 1):
        rewards[s] = -1
    else:
        rewards[s] = 0

# 'Gamma': the discount factor
gamma = 0.9

# ---- Environment ----

env = {} # {(state, action): (next_state, reward)}
for s in states:
    for a in actions:
        if a == 'U':
            next_state = (s[0], max(s[1]-1, 0))
        elif a == 'D':
            next_state = (s[0], min(s[1]+1, 2))
        elif a == 'L':
            next_state = (max(s[0]-1, 0), s[1])
        elif a == 'R':
            next_state = (min(s[0]+1, 3), s[1])

        if next_state == (1, 1): # Rock
            next_state = s

        env[(s, a)] = (next_state, rewards[s])

# ---- Agent ----

# A map from state to action
#policy = {s: np.random.choice(actions[s]) for s in states if actions[s]}

# The initial expected reward from state s
value = {s: 0 for s in states}

for _ in range(1):
    old_v = value
    for s in states:
        max_v = -np.inf

        for a in actions:
            next_state, reward = env[(s, a)]
            v = reward + (gamma * value[next_state])
            #print(s, a, value[next_state])

            if v > max_v:
                max_v = v
                #policy[s] = a

        value[s] = max_v

    for i, s in enumerate(states):
        if i % 4 == 0:
            print('\n'+'-'*35)
        print('%6.2f' % value[s], end=' | ')

    print('\n'+'-'*35)
    print('\n\n\n')

