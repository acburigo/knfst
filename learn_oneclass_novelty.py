import numpy as np
from knfst import calculate_knfst

def learn_oneclass_novelty(K):
    '''
    Compute one-class KNFST model by separating target data from origin in feature space

    INPUT
      K: (n x n) kernel matrix containing pairwise similarities between n training samples

    OUTPUT
      proj: Projection of KNFST
      target_value: The projections of training data into the null space
    '''

    n = K.shape[0]
    K = np.pad(K, pad_width=((0, 1), (0, 1)), mode='constant')

    labels = np.ones((n+1, 1))
    labels[n, 0] = 0

    proj = calculate_knfst(K, labels)
    target_value = np.mean(K[(labels == 1).reshape(-1), :].dot(proj), axis=0).reshape(-1, 1)
    proj = proj[0:n, :]

    #print K.shape, labels.shape, proj.shape, target_value.shape

    return proj, target_value
