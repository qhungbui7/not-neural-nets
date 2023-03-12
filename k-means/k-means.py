import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

seed = 42

### Description: generate synthetic data

def generate_data(means, cov, N, K):
    np.random.seed(seed) 
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X2 = np.random.multivariate_normal(means[2], cov, N)

    X = np.concatenate((X0, X1, X2), axis = 0)
    original_label = np.asarray([0]*N + [1]*N + [2]*N).T

    return X, original_label


# visualize data 
def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()

def kmeans_init_centers(X, k):
    np.random.seed(seed)

    # idxs = []
    # while len(idxs) < k:
    #     rand = np.random.randint(low=0, high=len(X))
    #     while rand in idxs: 
    #         rand = np.random.randint(low=0, high=len(X))
    #     idxs.append(rand)
    # return X[idxs]
    
    return X[np.random.choice(range(0, len(X)), size=k, replace=True, p=None)]


def kmeans_assign_labels(X, centers):
    return np.argmin(cdist(X, centers), axis=1)

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        centers[k] = np.average(X[labels == k], axis=0)
    return centers

def has_converged(centers, new_centers):
    return (centers == new_centers).all()


def kmeans(X, K):
    # save the center coordinates of each iteration
    centers = [kmeans_init_centers(X, K)]  
    # save the labels of each iteration
    labels = []
    it = 0 
    while True:
        # at each iteration:
        # 1. assign label for each points and append to labels
        # 2. update the centers
        # 3. check the convergence condition
        #    and append NEW center coordinates to centers
        # 4. update iteration 


        #1.
        assigned_labels = kmeans_assign_labels(X, centers[-1])
        labels.append(assigned_labels)
        
        #2.
        new_centers = kmeans_update_centers(X, assigned_labels, K)
        centers.append(new_centers)

        #3.
        if has_converged(centers=centers[-2], new_centers=centers[-1]):
            break 
        
        #4.
        it+=1
        
    
    return (centers, labels, it)


if __name__ == '__main__':
    N, K = 500, 3
    means = [[2, 2], [7, 3], [3, 6]]
    cov = [[1, 0], [0, 1]]
    # label
    X, original_label = generate_data(means, cov, N, K)
    kmeans_display(X, original_label)


    # execute
    (centers, labels, it) = kmeans(X, K)
    print('Centers found by k-means algorithm:')
    print(centers[-1])
    print('='*60)

    kmeans_display(X, labels[-1])




