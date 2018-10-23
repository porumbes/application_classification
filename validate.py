import numpy as np
from rsub import *
from matplotlib import pyplot as plt

a = [float(aa) for aa in open('python_result').read().split()]
b = [float(bb) for bb in open('cuda_result').read().split()]

if np.allclose(a, b):
    print('PASSED')
else:
    print('FAILED w/ corr=%f' % np.corrcoef(a, b)[0, 1])
    # _ = plt.scatter(a, b, s=1)
    # show_plot()