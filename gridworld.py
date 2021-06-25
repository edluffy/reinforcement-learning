import numpy as np

# A Markov Decision Process: <S, A, P, R, Gamma>

# 'S': set of states with Markov property
states = ((0, 0), (1, 0), (2, 0), (3, 0),
          (0, 1), (1, 1), (2, 1), (3, 1),
          (0, 2), (1, 2), (2, 2), (3, 2))


# ---- Environment ----

# 'A': set of actions agent can take when in state s
actions = {}
for s in states:
    actions[s] = ['U', 'D', 'L', 'R']
    if s[1] == 0:
        actions[s].remove('U')
    if s[1] == 2:
        actions[s].remove('D')
    if s[0] == 0:
        actions[s].remove('L')
    if s[0] == 3:
        actions[s].remove('R')
    actions[s] = tuple(actions[s])

# 'P': state transition probability matrix
transitions = None # Leave blank if fully deterministic

# 'R': gives immediate expected reward from state s
rewards = {}
for s in states:
    if s == (3, 0):
        rewards[s] = 1
    if s == (3, 1):
        rewards[s] = -1
    else:
        rewards[s] = 0

# 'Gamma': the discount factor
gamma = 0.99


# ---- Agent ----

# A map from state to action
policy = {}
for s in states:
    policy[s] = np.random.choice(actions[s])

# The initial expected reward from state s
value = {}
for s in states:
    if s == (3, 0) or s == (3, 1):
        value[s] = 0
    else:
        value[s] = -1

for s in states:
    for a in actions[s]:
        v = rewards[s] + (gamma * value[s])
        print(v)


