#!/usr/bin/env python
# author: Florian Kreyssig flk24@cam.ac.uk
# Some of this code was taken from https://stackoverflow.com/a/5551499
"""
    Class CustKMeans performs KMeans clustering
    Any distance measure from scipy.spatial.distance can be used
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from timeit import default_timer as timer
from pdb import set_trace as bp
from scipy.stats import multivariate_normal
import random

__metaclass__ = type

def k_means(X, n_clusters, init=None, tol=.001,
            max_iter=10, custom_dist="euclidean", p=2):
    """
    X : array-like, shape (n_samples, n_features)
        The observations to cluster.
    n_clusters : int
        The number of clusters to form
    init: ndarray, shaped (n_clusters, n_features), gives initial centroids
    max_iter : int, optional, default 10
        Maximum number of iterations of the k-means algorithm to run.
    tol : float, optional
        The relative increment in the results before declaring convergence.
    custom_dist: : str or callable, optional
        The distance metric to use.  If a string, the distance function can be
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
        'wminkowski', 'yule'.
    p : scalar, optional
        The p-norm to apply (for Minkowski, weighted and unweighted)
    """
    def init_centroids():
        """Compute the initial centroids"""
        n_samples = X.shape[0]
        init_n_samples = max(2*np.sqrt(n_samples), 10*n_clusters)
        _X = np.random.choice(np.arange(X.shape[0]),
                              size=init_n_samples, replace=False)
        _init = np.random.choice(np.arange(X.shape[0]),
                                 size=n_clusters, replace=False)
        return k_means(
            _X, _init, max_iter=max_iter, custom_dist=custom_dist)[0]

    n_samples, n_features = X.shape
    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)
    if n_samples < n_clusters:
        raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
            n_samples, n_clusters))
    if init is None:
        centres = init_centroids()
    else:
        n_centres, c_n_features = init.shape
        if n_centres != n_clusters:
            raise ValueError('The shape of the initial centers (%s)'
                             'does not match the number of clusters %d'
                             % (str(init.shape), n_clusters))
        if n_features != c_n_features:
            raise ValueError(
                "The number of features of the initial centers %d"
                "does not match the number of features of the data %d."
                % (c_n_features, n_features))
        centres = init.copy()

    sample_ids = np.arange(n_samples)
    prev_mean_dist = 0
    for iter_idx in range(1, max_iter+1):
        dist_to_all_centres = cdist(X, centres, metric=custom_dist, p=p)
        labels = dist_to_all_centres.argmin(axis=1)
        distances = dist_to_all_centres[sample_ids, labels]
        mean_distance = np.mean(distances)
        if (1 - tol) * prev_mean_dist <= mean_distance <= prev_mean_dist \
                or iter_idx == max_iter:
            break
        prev_mean_dist = mean_distance
        for each_center in range(n_centres):
            each_center_samples = np.where(labels == each_center)[0]
            if each_center_samples.any():
                centres[each_center] = np.mean(X[each_center_samples], axis=0)
    return centres, labels, distances


class CustKmeans:
    """
        Class to perform KMeans clustering.
        Can be used similar to sklearn.cluster Kmeans
    """

    def __init__(self, n_clusters=0, init=None, max_iter=10,
                 custom_dist='euclidean'):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.custom_dist = custom_dist
        self.centres = None

    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample."""
        return k_means(X, self.n_clusters, init=self.init,
                       max_iter=self.max_iter, custom_dist=self.custom_dist)

class FuzzyCMeans(object):
    ''' class provides Fuzzy C Means Algorithm for clustering n data points in d dimensions, 
    (depending on the number of clusters). 
    
    references: 
        Bezdek, Ehrlich, and Full. "FCM: The fuzzy c-means clustering algorithm." 1984.
    '''

    def __init__(self,data,c,m = 2, init = None, epsilon=1e-02, maxiter=30, metric='euclidean',verbose=False):
        '''
        Args:
            data: (n,d)-shaped d-dimensional ndarray objects containing float/integer data to be clustered
            c: integer, number of clusters 2<=c<=number of data points
            m: weight exponent 1<=m, large m -> fuzzier clustering, m=1 -> crisp partitioning
            epsilon: small, positive value, if norm of difference of successive membership matrices is smaller than epsilon, the iteration stops
            maxiter: integer, maximum number of iterations
            metric:  metric used to compute distances. for possible arguments see metric arguments of scipy.spatial.distance.pdist       
        '''

        if type(data) is list:
            raise NotImplementedError('FuzzyCMeans is not list compatible yet')

        self._data = data
        self._c = c
        self._m = m 
        self._eps = epsilon
        self._maxiter = maxiter
        self._cluster_labels = None
        self._cluster_centers = None
        self._cluster_dist = None
        self._metric = metric
        self._verbose = verbose
        self._membership = None
        self._iter = None
        self._time = None
        self.init = init

        if self._metric != 'euclidean':
            print('Initialized with %s metric. Use euclidean metric for classic Mean shift algorithm. \n'
                  'Bad things might happen, depending on your dataset and used metric.'%metric)

    def fit_predict(self):
        '''
        Runs the clustering iteration on the data it was given when initialized.     
        '''
        
        start_time = timer()
        
        #initialize
        [n,d] = self._data.shape
        # bp()
        if self.init is None:
            #initial membership matrix
            randommatrix = np.random.random((self._c, n)) 
            #normalize s.t. column sum = 1
            Uk = np.dot(randommatrix,np.diag(1/np.sum(randommatrix, axis=0))) 
            #row sum >0 for all rows
            while (np.sum(Uk, axis=1)<=0).any(): 
                randommatrix = np.random.random((self._c, n))
                Uk = np.dot(randommatrix,np.diag(1/np.sum(randommatrix, axis=0)))

            # index = random.sample(range(0,n),self._c)
            # self.init = self._data[index]
            # D = distance.cdist(self.init, self._data, metric = self._metric) 
            # Uk = np.power(D,- 2. / (self._m - 1))
            # Uk /=np.dot(np.ones((self._c, 1)),[Uk.sum(axis=0)])

        else:
            D = distance.cdist(self.init, self._data, metric = self._metric) 
            Uk = np.power(D,- 2. / (self._m - 1))
            Uk /=np.dot(np.ones((self._c, 1)),[Uk.sum(axis=0)])
        
        # self._data_norm = self._data/np.linalg.norm(self._data, axis=1)[:, None]
        # bp()
        #iterate  
        for k in range(self._maxiter):
            Uk = np.dot(Uk,np.diag(1/np.sum(Uk, axis=0)))
            Uk_powerm = np.power(Uk,self._m)
            #compute cluster centers
            vk = np.dot(np.diag(1/np.sum(Uk_powerm,axis=1)),np.dot(Uk_powerm, self._data))
            # vk_norm = vk/np.linalg.norm(vk, axis=1)[:, None]
            #distance matrix
            D = distance.cdist(vk, self._data, metric = self._metric) 
            Ukplus1 = np.power(D,- 2. / (self._m - 1))
            Ukplus1 /=np.dot(np.ones((self._c, 1)),[Ukplus1.sum(axis=0)])
            # Ukplus1 = 0.9 * Ukplus1 + 0.1 * np.dot(vk_norm,self._data_norm.T)
            if np.linalg.norm(Uk-Ukplus1) <self._eps:
                break
            Uk = Ukplus1.copy()
                
        # bp()
        self._cluster_labels = np.argmax(Uk,axis=0)
        self._membership = Uk
        self._cluster_centers = vk
        self._cluster_dist = np.min(D,axis=0)
        self._iter = k
        elapsed_time = timer() - start_time
        self._time = elapsed_time
        if self._verbose: 
            print('Finished after ' + str(elapsed_time))
            print('%s iterations until termination.' % str(k))
            print('Max within cluster distance to center: %f'%np.max(self._cluster_dist))
            print('Mean within cluster distance to center: %f' %np.mean(self._cluster_dist))
            print('Sum of within cluster squared errors: %f' % np.sum(np.square(self._cluster_dist)))
            
class softKMeans(object):
    ''' class provides Fuzzy C Means Algorithm for clustering n data points in d dimensions, 
    (depending on the number of clusters). 
    
    references: 
        Bezdek, Ehrlich, and Full. "FCM: The fuzzy c-means clustering algorithm." 1984.
    '''

    def __init__(self,data,c, init = None, epsilon=1e-02, maxiter=30, metric='euclidean',verbose=False,initcov=None):
        '''
        Args:
            data: (n,d)-shaped d-dimensional ndarray objects containing float/integer data to be clustered
            c: integer, number of clusters 2<=c<=number of data points
            m: weight exponent 1<=m, large m -> fuzzier clustering, m=1 -> crisp partitioning
            epsilon: small, positive value, if norm of difference of successive membership matrices is smaller than epsilon, the iteration stops
            maxiter: integer, maximum number of iterations
            metric:  metric used to compute distances. for possible arguments see metric arguments of scipy.spatial.distance.pdist       
        '''

        if type(data) is list:
            raise NotImplementedError('FuzzyCMeans is not list compatible yet')
        
        self._data = data
        self._c = c
        
        self._eps = epsilon
        self._maxiter = maxiter
        self._cluster_labels = None
        self._cluster_centers = None
        self._cluster_dist = None
        self._metric = metric
        self._verbose = verbose
        self._membership = None
        self._iter = None
        self._time = None
        self.init = init
        self.initcov = initcov
        if self._metric != 'euclidean':
            print('Initialized with %s metric. Use euclidean metric for classic Mean shift algorithm. \n'
                  'Bad things might happen, depending on your dataset and used metric.'%metric)
    
    def initialize_centers(self):
        x = self._data
        num_k = self._c
        N, D = x.shape
        centers = np.zeros((num_k, D))
        used_idx = []
        for k in range(num_k):
            idx = np.random.choice(N)
            while idx in used_idx:
                idx = np.random.choice(N)
            used_idx.append(idx)
            centers[k] = x[idx]
        return centers
    def fit_predict(self):
        '''
        Runs the clustering iteration on the data it was given when initialized.     
        '''
        
        start_time = timer()
        
        #initialize
        [n,d] = self._data.shape
        
        if self.init is None:
            #initial membership matrix
            # randommatrix = np.random.random((self._c, n)) 
            # #normalize s.t. column sum = 1
            # Uk = np.dot(randommatrix,np.diag(1/np.sum(randommatrix, axis=0))) 
            # #row sum >0 for all rows
            # while (np.sum(Uk, axis=1)<=0).any(): 
            #     randommatrix = np.random.random((self._c, n))
            #     Uk = np.dot(randommatrix,np.diag(1/np.sum(randommatrix, axis=0)))
            self.init = self.initialize_centers()
        # else:
        Uk = np.zeros((self._c,n))
        for ci in range(self._c):
            if self.initcov is not None:
                cur_cov = self.initcov
            else:
                cur_cov = np.eye(self._c)
            Uk[ci] = multivariate_normal.pdf(self._data,self.init[ci],cur_cov)
        
        Uk /=np.dot(np.ones((self._c, 1)),[Uk.sum(axis=0)])

        
        # self._data_norm = self._data/np.linalg.norm(self._data, axis=1)[:, None]
        # bp()
        # Uk[np.argmax(Uk,axis=0),np.arange(n)]=1
        # Uk[Uk<1] = 0
        
        #iterate  
        Ukplus1 = Uk.copy()
        for k in range(self._maxiter):
            Uk = np.dot(Uk,np.diag(1/np.sum(Uk, axis=0)))
            # Uk_powerm = np.power(Uk,self._m)
            #compute cluster centers
            vk = np.dot(np.diag(1/np.sum(Uk,axis=1)),np.dot(Uk, self._data))
            # vk_norm = vk/np.linalg.norm(vk, axis=1)[:, None]
            
            for ci in range(self._c):
                if self.initcov is not None:
                    cur_cov = self.initcov
                else:
                    cur_cov = np.eye(self._c)
                    
                Ukplus1[ci] = multivariate_normal.pdf(self._data,vk[ci],cur_cov)
                
            Ukplus1 /=np.dot(np.ones((self._c, 1)),[Ukplus1.sum(axis=0)])
            # Ukplus1 = 0.9 * Ukplus1 + 0.1 * np.dot(vk_norm,self._data_norm.T)
            # bp()
            if np.linalg.norm(Uk-Ukplus1) <self._eps:
                break
            Uk = Ukplus1.copy()
        
        # bp()
        self._cluster_labels = np.argmax(Uk,axis=0)
        self._membership = Uk
        self._cluster_centers = vk
        # self._cluster_dist = np.min(D,axis=0)
        self._iter = k
        elapsed_time = timer() - start_time
        self._time = elapsed_time
        if self._verbose: 
            print('Finished after ' + str(elapsed_time))
            print('%s iterations until termination.' % str(k))
            print('Max within cluster distance to center: %f'%np.max(self._cluster_dist))
            print('Mean within cluster distance to center: %f' %np.mean(self._cluster_dist))
            print('Sum of within cluster squared errors: %f' % np.sum(np.square(self._cluster_dist)))

    def predict(self,X):

        [n,d] = X.shape
        Uk = np.zeros((self._c,n))
        
        for ci in range(self._c):
            if self.initcov is not None:
                cur_cov = self.initcov
            else:
                cur_cov = np.eye(self._c)
            Uk[ci] = multivariate_normal.pdf(X,self._cluster_centers[ci],cur_cov)
        
        Uk /=np.dot(np.ones((self._c, 1)),[Uk.sum(axis=0)])
        labels = np.argmax(Uk,axis=0)
        return labels, Uk



class softKMeans_v2(object):
    ''' class provides Fuzzy C Means Algorithm for clustering n data points in d dimensions, 
    (depending on the number of clusters). 
    
   '''

    def __init__(self,data,c, init = None, epsilon=1e-02, maxiter=30, metric='euclidean',verbose=False,beta=1.):
        '''
        Args:
            data: (n,d)-shaped d-dimensional ndarray objects containing float/integer data to be clustered
            c: integer, number of clusters 2<=c<=number of data points
            m: weight exponent 1<=m, large m -> fuzzier clustering, m=1 -> crisp partitioning
            epsilon: small, positive value, if norm of difference of successive membership matrices is smaller than epsilon, the iteration stops
            maxiter: integer, maximum number of iterations
            metric:  metric used to compute distances. for possible arguments see metric arguments of scipy.spatial.distance.pdist       
        '''

        if type(data) is list:
            raise NotImplementedError('FuzzyCMeans is not list compatible yet')

        self._data = data
        self._c = c
        
        self._eps = epsilon
        self._maxiter = maxiter
        self._cluster_labels = None
        self._cluster_centers = None
        self._cluster_dist = None
        self._metric = metric
        self._verbose = verbose
        self._membership = None
        self._iter = None
        self._time = None
        self.init = init
        self.beta = beta

        if self._metric != 'euclidean':
            print('Initialized with %s metric. Use euclidean metric for classic Mean shift algorithm. \n'
                  'Bad things might happen, depending on your dataset and used metric.'%metric)
    
    def initialize_centers(self):
        x = self._data
        num_k = self._c
        N, D = x.shape
        centers = np.zeros((num_k, D))
        used_idx = []
        for k in range(num_k):
            idx = np.random.choice(N)
            while idx in used_idx:
                idx = np.random.choice(N)
            used_idx.append(idx)
            centers[k] = x[idx]
        return centers
    
    def update_centers(self, r):
        N, D = self._data.shape
        centers = np.zeros((self._c, D))
        for k in range(self._c):
            centers[k] = r[:, k].dot(self._data) / r[:, k].sum()
        return centers
    
    def square_dist(a, b):
        return (a - b) ** 2
    
    def cost_func(self,r,centers):
        
        cost = 0
        for k in range(self._c):
            norm = np.linalg.norm(self._data - centers[k], 2)
            cost += (norm * np.expand_dims(r[:, k], axis=1) ).sum()
        return cost
    
    
    def cluster_responsibilities(self,centers):
        N, _ = self._data.shape
        K, D = centers.shape
        R = np.zeros((N, K))
    
        for n in range(N):        
            # bp()
            R[n] = np.exp(-self.beta * np.linalg.norm(centers - self._data[n], 2, axis=1)) 
        R /= R.sum(axis=1, keepdims=True)
    
        return R
    
    def fit_predict(self):
        '''
        Runs the clustering iteration on the data it was given when initialized.     
        '''
        
        start_time = timer()
        
        #initialize
        [n,d] = self._data.shape
        if self.init is None:
            centers = self.initialize_centers()
        else:
            centers = self.init
        prev_cost = 0
        for k in range(self._maxiter):
            Uk = self.cluster_responsibilities(centers)
            centers = self.update_centers( Uk)
            cost = self.cost_func(Uk, centers)
            # print('cost: ',cost)
            if np.abs(cost - prev_cost) < 1e-5:
                break
            prev_cost = cost
        Uk = Uk.T
        # bp()
        self._cluster_labels = np.argmax(Uk,axis=0)
        self._membership = Uk
        self._cluster_centers = centers
        # self._cluster_dist = np.min(D,axis=0)
        self._iter = k
        elapsed_time = timer() - start_time
        self._time = elapsed_time
       