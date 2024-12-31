import numpy as np
from pydra.design import python

NumpyCorrelate = python.define(np.correlate)

numpy_correlate = NumpyCorrelate(a=[1, 2, 3], v=[0, 1, 0.5])

outputs = numpy_correlate()

print(outputs.out)
