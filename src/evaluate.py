import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

epsilon = 1e-10
def evaluate_score(args, outputs, targets):
    threshold = 0.3    #0.03*10
    
    Error = 0
    for i in range(96):
        if targets[i]>threshold:
            Error += np.abs(targets[i] - outputs[i])/outputs[i]
        else:
            pass
    Error /= 96



    return score


