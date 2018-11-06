import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a = [1,2,2,1]
b = pd.Series(a, index=['up', 'down', 'left', 'right'])
print(b[b == np.max(b)].index)
