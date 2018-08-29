from typing import List

import numpy as np
from math import pow



class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        #print("max iterations: ", self.max_iter)
        N, D = x.shape
        np.random.seed(42)
        #N, D = x.shape
        u = []
        J_old = 0
        for i in range(self.n_cluster):
            ran = np.random.randint(N)
            u.append(x[ran])
        u = np.array(u)
        K = u.shape[0]

        for i in range(self.max_iter):
            assignment = [0 for _ in range(N)]
            for j in range(N):
                min_dist = np.linalg.norm(x[j] - u[0])**2
                assignment[j] = 0
                for k in range(1,K):
                    a = np.linalg.norm(x[j] - u[k])**2
                    if (a < min_dist):
                        min_dist = a
                        assignment[j] = k
            J_new = 0
            for n_idx in range(N):
                for k in range(K):
                    if (assignment[n_idx]==k):
                        J_new = J_new + np.linalg.norm(x[n_idx] - u[k])**2
            J_new = J_new/N

            if (abs(J_new - J_old) < self.e):
                break
            J_old = J_new
            
            for k in range (K): 
                first = np.zeros(D)
                second = 0.0
                for l in range(N):
                    if(k == assignment[l]):
                        first += x[l]
                        second +=1.0
                first = first/second
                u[k] = first


        return np.asarray(u), np.array(assignment), i+1


            






        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE


        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE

        #for i in y:
            #print("i: ", i)

        centroid_labels = []
        k_means = KMeans(n_cluster=self.n_cluster, max_iter=100, e=1e-6)
        #print("n_cluster: ", self.n_cluster)
        centroids, membership, iters = k_means.fit(x)
        bins = []
        #print("centroid[0]: ", centroids[0])
        # for i in range (len(centroids)):
        #     sublist = []
        #     bins.append(sublist)
        #print("len membership: ", len(membership))
        bins = [[] for i in centroids]

        for i in range (len(membership)):
            bins[membership[i]].append(y[i])

        for a in bins:
            centroid_labels.append(np.bincount(a).argmax())


        centroid_labels = np.asarray(centroid_labels)


        # DONOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''



        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        y = []
        for i in x:
            distances = []
            for a in range(len(self.centroids)):
                dist = np.linalg.norm(i - self.centroids[a])**2
                distances.append(dist)


            y.append(self.centroid_labels[np.argmin(distances)])
            #print(self.centroid_labels[a])
        return np.asarray(y)

        # DONOT CHANGE CODE BELOW THIS LINE
