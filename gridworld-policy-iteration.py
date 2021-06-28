import numpy as np
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
gamma = 1

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

def eval_policy():
    while True:
        delta = 0
        for state in states:
            old_value = value[state]
            # pi(a|s) = 0.25, p(s',r|s,a) = 1
            value[state] = 0.25*sum(r+gamma*value[ns] for (s, a), (ns, r) in env.items() if s == state)
            delta = max(delta, abs(old_value-value[state]))
        if delta < 0.0001:
            return

def improve_policy():
    for state in states:
        old_action = policy[state]
        # p(s',r|s,a) = 1
        action_values = {a: r+gamma*value[ns] for (s, a), (ns, r) in env.items() if s == state}
        policy[state] = max(action_values, key=action_values.get)

        if old_action != policy[state]:
            return False
    return True

def print_grid(var):
    if var == value:
        fstr = '%6.2f'
    else:
        fstr = '%6c'

    for i, s in enumerate(states):
        if i % 4 == 0:
            print('\n'+'-'*35)
        print(fstr % var[s], end=' | ')
    print('\n'+'-'*35)
    print('\n\n\n')

i = 0
policy_stable = False
while not policy_stable:
    print('Iteration: ', i)
    eval_policy()
    policy_stable = improve_policy()

    print_grid(value)
    print_grid(policy)
    i+=1
