import numpy as np
def J(targ:np.ndarray, real:np.ndarray) -> float:
    diff = targ - real
    return np.dot(diff,diff)