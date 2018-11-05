# 不涉及神经网路，暂时不用tensorflow。
import numpy as np
import pandas as pd

# the key of q-learning code:
# learn rate, gama(discount?), q_table, reward, e_greedy(choice action), action

class qlearing:
    def __init__(self, action, lr=0.1, e_greedy=0.9, gama=0.9):
        self.lr = lr
        self.e_greedy = e_greedy
        self.gama = gama
        self.action = action
        self.q_table = pd.DataFrame(columns=action, dtype=np.float64)

    def choose_action(self, state):
        self.check_exit(state)
        if np.    
