import numpy as np
from rsub import *
from matplotlib import pyplot as plt

a = [float(aa) for aa in open('tmp').read().split()]
b = [float(bb) for bb in open('python_result_test').read().split()]

if np.allclose(a, b):
    print('PASSED')
else:
    for i, (aa, bb) in enumerate(zip(a, b)):
        if not np.allclose(aa, bb):
            print(i, aa, bb)
    
    print('FAILED w/ corr=%f' % np.corrcoef(a, b)[0, 1])
    # _ = plt.scatter(a, b, s=1)
    # show_plot()