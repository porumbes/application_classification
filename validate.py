import numpy as np

a = [float(aa) for aa in open('python_result').read().split()]
b = [float(bb) for bb in open('orig_result').read().split()]
assert np.allclose(a, b)
print('PASSED')