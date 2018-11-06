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
        if(np.random.rand() < self.e_greedy):
            find_res = self.q_table.loc[state, :]
            res_action = np.random.choice(find_res[find_res == np.max(find_res)].index)
        else:
            res_action = np.random.choice(self.action)
        return res_action

    def check_exit(self, state):
        if state not in self.q_table.index:
            new_series = pd.Series([0]*len(self.action), index=self.action, name=state)
            self.q_table = self.q_table.append(new_series)

    def learn(self, s, a, r, s_):
        # loss = real - predict, real = r + gama*max(next_state), predict = q_table
        # update method: new_q_table = old_q_table + lr*loss
        self.check_exit(s_)
        if(s != 'terminal'):
            q_real = r + self.gama * np.max(self.q_table.loc[s_, :])
        else:
            q_real = r
        q_predict = self.q_table.loc[s, a]
        loss = q_real - q_predict
        self.q_table.loc[s, a] += self.lr*loss
