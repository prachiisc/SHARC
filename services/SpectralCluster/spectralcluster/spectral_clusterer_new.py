from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.cluster import KMeans
#from spectralcluster import refinement
# from spectralcluster import utils
from . import refinement
from . import utils
from . import custom_distance_kmeans
# import refinement
# import utils
# import custom_distance_kmeans
from pdb import set_trace as bp
from matplotlib import pyplot as plt
from scipy.sparse.csgraph import laplacian
import matplotlib.pyplot as plt
import matplotlib as mtb
mtb.use('Agg')

# DEFAULT_REFINEMENT_SEQUENCE = [
#     "CropDiagonal",
#     "GaussianBlur",
#     "RowWiseThreshold",
#     "Symmetrize",
#     "Diffuse",
#     "RowWiseNormalize",
# ]

# DEFAULT_REFINEMENT_SEQUENCE = [
#     "CropDiagonal",
#     "Symmetrize",
#     "Diffuse",
#     "RowWiseNormalize",
# ]

DEFAULT_REFINEMENT_SEQUENCE = [
    "CropDiagonal",
]
# DEFAULT_REFINEMENT_SEQUENCE = []


class SpectralClusterer(object):
    def __init__(
            self,
            min_clusters=None,
            max_clusters=None,
            gaussian_blur_sigma=1,
            p_percentile=0.95,
            thresholding_soft_multiplier=0.01,
            stop_eigenvalue=1e-2,
            custom_dist=None,
            custom_dist_maxiter=(2, 10),
            refinement_sequence=DEFAULT_REFINEMENT_SEQUENCE):
        """Constructor of the clusterer.
        Args:
            min_clusters: minimal number of clusters allowed (only effective
                if not None)
            max_clusters: maximal number of clusters allowed (only effective
                if not None), can be used together with min_clusters to fix
                the number of clusters
            gaussian_blur_sigma: sigma value of the Gaussian blur operation
            p_percentile: the p-percentile for the row wise thresholding
            thresholding_soft_multiplier: the multiplier for soft threhsold,
                if this value is 0, then it's a hard thresholding
            stop_eigenvalue: when computing the number of clusters using
                Eigen Gap, we do not look at eigen values smaller than this
                value
            custom_dist: custome distance for KMeans e.g. cosine. Any distance
                of scipy.spatial.distance can be used
            custom_dist_maxiter: int or tuple,
                if int then number of iterations
                if tuple then tuple[0] is number of KMeans++ iterations
            refinement_sequence: a list of strings for the sequence of
                refinement operations to apply on the affinity matrix
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.p_percentile = p_percentile
        self.thresholding_soft_multiplier = thresholding_soft_multiplier
        self.stop_eigenvalue = stop_eigenvalue
        self.custom_dist = custom_dist
        if isinstance(custom_dist_maxiter, int):
            self.custom_dist_maxiter = (0, custom_dist_maxiter)
        elif isinstance(custom_dist_maxiter, tuple):
            self.custom_dist_maxiter = custom_dist_maxiter
        else:
            raise ValueError(
                "custom_dist_maxiter has to be either int or tuple,"
                "received type=%s" % str(type(custom_dist_maxiter)))
        self.refinement_sequence = refinement_sequence

    def _get_refinement_operator(self, name):
        """Get the refinement operator.
        Args:
            name: operator class name as a string
        Returns:
            object of the operator
        Raises:
            ValueError: if name is an unknown refinement operation
        """
        if name == "CropDiagonal":
            return refinement.CropDiagonal()
        elif name == "GaussianBlur":
            return refinement.GaussianBlur(self.gaussian_blur_sigma)
        elif name == "RowWiseThreshold":
            return refinement.RowWiseThreshold(
                self.p_percentile,
                self.thresholding_soft_multiplier)
        elif name == "Symmetrize":
            return refinement.Symmetrize()
        elif name == "Diffuse":
            return refinement.Diffuse()
        elif name == "RowWiseNormalize":
            return refinement.RowWiseNormalize()
        else:
            raise ValueError("Unknown refinement operation: {}".format(name))


    def refinementonly(self,X,cosine):

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        # bp()
        # plt.figure();
        # plt.imshow(X)
        #  Compute affinity matrix.
        if cosine:
            affinity = utils.compute_affinity_matrix(X)
        else:
            affinity = X
        # Refinement opertions on the affinity matrix.
        for refinement_name in self.refinement_sequence:
            op = self._get_refinement_operator(refinement_name)
            affinity = op.refine(affinity)
            # plt.figure()
            # plt.imshow(affinity)
            # plt.title(refinement_name)
        return affinity

    def predict(self, X):
        """Perform spectral clustering on data X.
        Args:
            X: numpy array of shape (n_samples, n_features)
        Returns:
            labels: numpy array of shape (n_samples,)
        Raises:
            TypeError: if X has wrong type
            ValueError: if X has wrong shape
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        #  Compute affinity matrix.
        affinity = utils.compute_affinity_matrix(X)

        # Refinement opertions on the affinity matrix.
        for refinement_name in self.refinement_sequence:
            op = self._get_refinement_operator(refinement_name)
            affinity = op.refine(affinity)

        # Perform eigen decomposion.
        (eigenvalues, eigenvectors) = utils.compute_sorted_eigenvectors(
            affinity)
        # Get number of clusters.
        k = utils.compute_number_of_clusters(
            eigenvalues, self.max_clusters, self.stop_eigenvalue)
        if self.min_clusters is not None:
            k = max(k, self.min_clusters)

        # Get spectral embeddings.
        spectral_embeddings = eigenvectors[:, :k]

        # Run K-Means
        # Using custom_dist a custom distance measure can be used
        # Setting custom_dist=cosine is consistent with the paper
        # For custom_dist=None KMeans++ algorithm is used
        if self.custom_dist is None:
            kmeans_clusterer = KMeans(
                n_clusters=k,
                init="k-means++",
                max_iter=300,
                random_state=0)
        else:
            if self.custom_dist_maxiter[0] > 0:
                kmeans_clusterer = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    max_iter=self.custom_dist_maxiter[0],
                    random_state=0)
                kmeans_clusterer.fit(spectral_embeddings)
                init = kmeans_clusterer.cluster_centers_
            else:
                init = None
            kmeans_clusterer = custom_distance_kmeans.CustKmeans(
                n_clusters=k,
                init=init,
                max_iter=self.custom_dist_maxiter[1],
                custom_dist=self.custom_dist)
        labels = kmeans_clusterer.fit_predict(spectral_embeddings)
        return labels


    def predict_withscores(self, X):
        """Perform spectral clustering on score matrix X.
        Args:
            X: numpy array of shape (n_samples, n_samples)
        Returns:
            labels: numpy array of shape (n_samples,)
        Raises:
            TypeError: if X has wrong type
            ValueError: if X has wrong shape
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        #  Compute affinity matrix.
        # affinity = utils.compute_affinity_matrix(X)
        affinity = X.copy()
        # bp()
        
        # Refinement opertions on the affinity matrix.
        for refinement_name in self.refinement_sequence:
            op = self._get_refinement_operator(refinement_name)
            affinity = op.refine(affinity)
        # avg_val = (affinity.max()+affinity.min())/2
        # affinity[affinity>=avg_val] = 1.0
        # Perform eigen decomposion.
        (eigenvalues, eigenvectors) = utils.compute_sorted_eigenvectors(
            affinity)
        
        # Get number of clusters.
        stop_eigenvalue = self.stop_eigenvalue
        print('stop_eigenvalue ratio:',stop_eigenvalue)
        if self.max_clusters == self.min_clusters:
            k= self.max_clusters
        else:
            k = utils.compute_number_of_clusters(
                eigenvalues, self.max_clusters, stop_eigenvalue)
            if self.min_clusters is not None:
                k = max(k, self.min_clusters)
        print('k:',k)
        # bp()
        # Get spectral embeddings.
        spectral_embeddings = eigenvectors[:, :k]
        
        # spectral_embeddings = eigenvectors[:, :10]

        # Run K-Means
        # Using custom_dist a custom distance measure can be used
        # Setting custom_dist=cosine is consistent with the paper
        # For custom_dist=None KMeans++ algorithm is used
        if self.custom_dist is None:
            kmeans_clusterer = KMeans(
                n_clusters=k,
                init="k-means++",
                max_iter=300,
                random_state=0)
            kmeans_clusterer.fit(spectral_embeddings)
            init = kmeans_clusterer.cluster_centers_
            print('xvec dim: {} init: {}'.format(spectral_embeddings.shape,init.shape))
            # bp()
            # plt.figure()
            # plt.scatter(spectral_embeddings[:,0],spectral_embeddings[:,1])
            # plt.scatter(init[:,0],init[:,1], marker='x')
            # plt.savefig('scatterplda1.png')
             
        else:
            if self.custom_dist_maxiter[0] > 0:
                kmeans_clusterer = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    max_iter=self.custom_dist_maxiter[0],
                    random_state=0)
                kmeans_clusterer.fit(spectral_embeddings)
                init = kmeans_clusterer.cluster_centers_
            else:
                init = None
            kmeans_clusterer = custom_distance_kmeans.CustKmeans(
                n_clusters=k,
                init=init,
                max_iter=self.custom_dist_maxiter[1],
                custom_dist=self.custom_dist)
        labels = kmeans_clusterer.fit_predict(spectral_embeddings)
        labels = labels[1]
        
        return labels


    def predict_withscores_laplacian(self, X):
        """Perform spectral clustering on score matrix X.
        https://github.com/cvqluu/nn-similarity-diarization/blob/master/cluster.py
        Args:
            X: numpy array of shape (n_samples, n_samples)
        Returns:
            labels: numpy array of shape (n_samples,)
        Raises:
            TypeError: if X has wrong type
            ValueError: if X has wrong shape
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        #  Compute affinity matrix.
        # affinity = utils.compute_affinity_matrix(X)
        affinity = X.copy()
        # Refinement opertions on the affinity matrix.
        for refinement_name in self.refinement_sequence:
            op = self._get_refinement_operator(refinement_name)
            affinity = op.refine(affinity)

        np.fill_diagonal(affinity, 0.)
        L_norm = laplacian(affinity, normed=True)
        eigvals, eigvecs = np.linalg.eig(L_norm)
        index_arr = np.argsort(np.real(eigvals))
        eigvals = eigvals[index_arr]
        eigvecs = eigvecs[:,index_arr]
     
        if self.max_clusters == self.min_clusters:
            # P = np.real(eigvecs).T[:self.max_clusters].T           
            P = np.real(eigvecs)[:,:self.max_clusters]
            eigvals_min = eigvals[:self.max_clusters]
            # P = P[:,eigvals_min>0]
        else:
            # bp()
            kmask1 = 0 < np.real(eigvals) 
            pos_eigen = np.real(eigvals)[kmask1]
            ratio = np.real(eigvals)/np.max(pos_eigen)
            kmask = ratio < self.stop_eigenvalue
            # kmask = kmask1*kmask2
            
            P = np.real(eigvecs)[:,kmask]
            clusters_count = min(self.max_clusters,P.shape[1])
            clusters_count = max(self.min_clusters,clusters_count)
            eigvals_min = eigvals[:clusters_count]
            P  = P[:,:clusters_count]
            # P = P[:,eigvals_min > 0]
        # bp()
        print('clusters_count:{}'.format(P.shape[1]))
        km = KMeans(n_clusters=P.shape[1])  
        
        labels = km.fit_predict(P)
        init = km.cluster_centers_
        # bp()
        # plt.figure()
        # plt.scatter(P[:,0],P[:,1])
        # plt.scatter(init[:,0],init[:,1], marker='x')
        # plt.savefig('scatter5.png')
        return labels
        
    def predict_with_cmeans(self, X):
        """Perform spectral clustering on score matrix X.
        Args:
            X: numpy array of shape (n_samples, n_samples)
        Returns:
            labels: numpy array of shape (n_samples,)
        Raises:
            TypeError: if X has wrong type
            ValueError: if X has wrong shape
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        #  Compute affinity matrix.
        # affinity = utils.compute_affinity_matrix(X)
        affinity = X.copy()
        # Refinement opertions on the affinity matrix.
        for refinement_name in self.refinement_sequence:
            op = self._get_refinement_operator(refinement_name)
            affinity = op.refine(affinity)

        # Perform eigen decomposion.
        (eigenvalues, eigenvectors) = utils.compute_sorted_eigenvectors(
            affinity)
        # Get number of clusters.
        stop_eigenvalue = self.stop_eigenvalue
        # bp()
        k = utils.compute_number_of_clusters(
            eigenvalues, self.max_clusters, stop_eigenvalue)
        if self.min_clusters is not None:
            k = max(k, self.min_clusters)
        # bp()
        # Get spectral embeddings.
        spectral_embeddings = eigenvectors[:, :k]

        # Run K-Means
        # Using custom_dist a custom distance measure can be used
        # Setting custom_dist=cosine is consistent with the paper
        # For custom_dist=None KMeans++ algorithm is used
        if self.custom_dist is None:
            kmeans_clusterer = custom_distance_kmeans.FuzzyCMeans(spectral_embeddings,
                c=k,
                m=2,
                init=None,
                maxiter=self.custom_dist_maxiter[1]
               )
        else:
            # if self.custom_dist_maxiter[0] > 0:
            #     kmeans_clusterer = KMeans(
            #         n_clusters=k,
            #         init="k-means++",
            #         max_iter=self.custom_dist_maxiter[0],
            #         random_state=0)
            #     kmeans_clusterer.fit(spectral_embeddings)
            #     init = kmeans_clusterer.cluster_centers_
            # else:
            #     init = None
            kmeans_clusterer = custom_distance_kmeans.FuzzyCMeans(spectral_embeddings,
                c=k,
                m=2,
                init=None,
                maxiter=self.custom_dist_maxiter[1],
                metric=self.custom_dist)
        kmeans_clusterer.fit_predict()
        
        labels = kmeans_clusterer._cluster_labels
        membership = kmeans_clusterer._membership
        # bp()
        return labels, membership.T

    def predict_with_softkmeans_modified(self, X, clean_ind):
        """Perform spectral clustering on score matrix X.
        Args:
            X: numpy array of shape (n_samples, n_samples)
        Returns:
            labels: numpy array of shape (n_samples,)
        Raises:
            TypeError: if X has wrong type
            ValueError: if X has wrong shape
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        #  Compute affinity matrix.
        # affinity = utils.compute_affinity_matrix(X)
        affinity = X.copy()
        # Refinement opertions on the affinity matrix.
        for refinement_name in self.refinement_sequence:
            op = self._get_refinement_operator(refinement_name)
            affinity = op.refine(affinity)

        # np.fill_diagonal(affinity, 0.)
        # L_norm = laplacian(affinity, normed=True)
        
        # N = affinity.shape[0]
        # L_norm  = np.eye(N) - L_norm

        # Perform eigen decomposion.
        (eigenvalues, eigenvectors) = utils.compute_sorted_eigenvectors(
            affinity)
        # Get number of clusters.
        stop_eigenvalue = self.stop_eigenvalue
        # bp()
        k = utils.compute_number_of_clusters(
            eigenvalues, self.max_clusters, stop_eigenvalue)
        if self.min_clusters is not None:
            k = max(k, self.min_clusters)

        print('k: ',k)
        # Get spectral embeddings.
        spectral_embeddings_org = eigenvectors[:, :k]
        # bp()
        spectral_embeddings = spectral_embeddings_org[clean_ind]

        if 1: #self.custom_dist is None:
            initkmeans_clusterer = KMeans(
                n_clusters=k,
                init="k-means++",
                max_iter=30,
                random_state=0)
            
            initkmeans_clusterer.fit(spectral_embeddings)
            init = initkmeans_clusterer.cluster_centers_
            dist = initkmeans_clusterer.inertia_
            initlabels = initkmeans_clusterer.labels_
            # initcov = []
            # for i in range(k):
            #     initcov.append(np.cov(spectral_embeddings[initlabels==i].T))
            # initcov = np.zeros((k,k))
            # for i in range(k):
            #     initcov +=np.cov(spectral_embeddings[initlabels==i].T)
            # initcov /=k
            if k >1:
                initcov = np.diag(np.diag(np.cov(spectral_embeddings.T)))
                # initcov = np.cov(spectral_embeddings.T)
                initcov = initcov + (np.eye(k)*1e-5)
            else:
                initcov = np.var(spectral_embeddings)
                
            # bp()
            kmeans_clusterer = custom_distance_kmeans.softKMeans(spectral_embeddings,
                c=k,
                init=init,
                initcov=initcov,
                maxiter=self.custom_dist_maxiter[1]
                )
        else:
            initkmeans_clusterer = custom_distance_kmeans.CustKmeans(
                n_clusters=k,
                init=init[0],
                max_iter=self.custom_dist_maxiter[1],
                custom_dist=self.custom_dist)
            
            init = initkmeans_clusterer.fit_predict(spectral_embeddings)
            kmeans_clusterer = custom_distance_kmeans.softKMeans(spectral_embeddings,
                c=k,
                init=init[0],
                maxiter=10,
                metric=self.custom_dist)
        kmeans_clusterer.fit_predict()
        # bp()
        labels, membership = kmeans_clusterer.predict(spectral_embeddings_org)
        # labels = kmeans_clusterer._cluster_labels
        # membership = kmeans_clusterer._membership
        
        return labels, membership.T
    
    def predict_with_softkmeans(self, X):
        """Perform spectral clustering on score matrix X.
        Args:
            X: numpy array of shape (n_samples, n_samples)
        Returns:
            labels: numpy array of shape (n_samples,)
        Raises:
            TypeError: if X has wrong type
            ValueError: if X has wrong shape
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        #  Compute affinity matrix.
        # affinity = utils.compute_affinity_matrix(X)
        affinity = X.copy()
        # Refinement opertions on the affinity matrix.
        for refinement_name in self.refinement_sequence:
            op = self._get_refinement_operator(refinement_name)
            affinity = op.refine(affinity)

        # np.fill_diagonal(affinity, 0.)
        # L_norm = laplacian(affinity, normed=True)
        
        # N = affinity.shape[0]
        # L_norm  = np.eye(N) - L_norm

        # Perform eigen decomposion.
        (eigenvalues, eigenvectors) = utils.compute_sorted_eigenvectors(
            affinity)
        # Get number of clusters.
        stop_eigenvalue = self.stop_eigenvalue
        # bp()
        if self.min_clusters == self.max_clusters:
            k = self.max_clusters
        else:
            k = utils.compute_number_of_clusters(
                eigenvalues, self.max_clusters, stop_eigenvalue)
            if self.min_clusters is not None:
                k = max(k, self.min_clusters)

        print('k: ',k)
        # Get spectral embeddings.
        spectral_embeddings = eigenvectors[:, :k]
        # bp()
        # Run K-Means
        # Using custom_dist a custom distance measure can be used
        # Setting custom_dist=cosine is consistent with the paper
        # For custom_dist=None KMeans++ algorithm is used
        # if self.custom_dist_maxiter[0] > 0:
        #         kmeans_clusterer = KMeans(
        #             n_clusters=k,
        #             init="k-means++",
        #             max_iter=self.custom_dist_maxiter[0],
        #             random_state=0)
        #         kmeans_clusterer.fit(spectral_embeddings)
        #         init = kmeans_clusterer.cluster_centers_
        # else:
        #         init = None
        if 1: #self.custom_dist is None:
            initkmeans_clusterer = KMeans(
                n_clusters=k,
                init="k-means++",
                max_iter=30,
                random_state=0)
            
            initkmeans_clusterer.fit(spectral_embeddings)
            init = initkmeans_clusterer.cluster_centers_
            dist = initkmeans_clusterer.inertia_
            initlabels = initkmeans_clusterer.labels_
            # initcov = []
            # for i in range(k):
            #     initcov.append(np.cov(spectral_embeddings[initlabels==i].T))
            # initcov = np.zeros((k,k))
            # for i in range(k):
            #     initcov +=np.cov(spectral_embeddings[initlabels==i].T)
            # initcov /=k
            if k >1:
                initcov = np.diag(np.diag(np.cov(spectral_embeddings.T)))
                # initcov = np.cov(spectral_embeddings.T)
                initcov = initcov + (np.eye(k)*1e-5)
                # bp()
                kmeans_clusterer = custom_distance_kmeans.softKMeans(spectral_embeddings,
                    c=k,
                    init=init,
                    initcov=initcov,
                    maxiter=self.custom_dist_maxiter[1]
                    )
            # else:
                # initcov = np.var(spectral_embeddings)
        else:
            initkmeans_clusterer = custom_distance_kmeans.CustKmeans(
                n_clusters=k,
                init=init[0],
                max_iter=self.custom_dist_maxiter[1],
                custom_dist=self.custom_dist)
            
            init = initkmeans_clusterer.fit_predict(spectral_embeddings)
            kmeans_clusterer = custom_distance_kmeans.softKMeans(spectral_embeddings,
                c=k,
                init=init[0],
                maxiter=10,
                metric=self.custom_dist)
        if k >1:
            kmeans_clusterer.fit_predict()
            # bp()
            labels = kmeans_clusterer._cluster_labels
            membership = kmeans_clusterer._membership
        else:
            labels = np.zeros((affinity.shape[0],),dtype=int)
            membership = np.ones((1,affinity.shape[0]))
        
        return labels, membership.T
  
    def predict_with_softkmeans_laplacian(self, X):
        """Perform spectral clustering on score matrix X.
        Args:
            X: numpy array of shape (n_samples, n_samples)
        Returns:
            labels: numpy array of shape (n_samples,)
        Raises:
            TypeError: if X has wrong type
            ValueError: if X has wrong shape
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        #  Compute affinity matrix.
        # affinity = utils.compute_affinity_matrix(X)
        affinity = X.copy()
        # Refinement opertions on the affinity matrix.
        for refinement_name in self.refinement_sequence:
            op = self._get_refinement_operator(refinement_name)
            affinity = op.refine(affinity)

        np.fill_diagonal(affinity, 0.)
        L_norm = laplacian(affinity, normed=True)
        

        eigvals, eigvecs = np.linalg.eig(L_norm)
        index_arr = np.argsort(np.real(eigvals))
        eigvals = eigvals[index_arr]
        eigvecs = eigvecs[:,index_arr]
        
        if self.max_clusters == self.min_clusters:
            # P = np.real(eigvecs).T[:self.max_clusters].T           
            spectral_embeddings = np.real(eigvecs)[:,:self.max_clusters]
            eigvals_min = eigvals[:self.max_clusters]
            # P = P[:,eigvals_min>0]
            k = self.max_clusters
        else:
            kmask = np.real(eigvals) < self.stop_eigenvalue
            P = np.real(eigvecs)[:,kmask]
            clusters_count = min(self.max_clusters,P.shape[1])
            clusters_count = max(self.min_clusters,clusters_count)
            eigvals_min = eigvals[:clusters_count]
            spectral_embeddings  = P[:,:clusters_count]
            k = clusters_count
        

        # Run K-Means
        # Using custom_dist a custom distance measure can be used
        # Setting custom_dist=cosine is consistent with the paper
        # For custom_dist=None KMeans++ algorithm is used
        
        # if self.custom_dist is None:
        initkmeans_clusterer = KMeans(
            n_clusters=k,
            init="k-means++",
            max_iter=30,
            random_state=0)
        
        initkmeans_clusterer.fit(spectral_embeddings)
        init = initkmeans_clusterer.cluster_centers_
        dist = initkmeans_clusterer.inertia_
        initlabels = initkmeans_clusterer.labels_
        
        if k >1:
            initcov = np.diag(np.diag(np.cov(spectral_embeddings.T)))
        else:
            initcov = np.var(spectral_embeddings)
        # bp()
        kmeans_clusterer = custom_distance_kmeans.softKMeans(spectral_embeddings,
            c=k,
            init=init,
            initcov=initcov,
            maxiter=self.custom_dist_maxiter[1]
            )
        
        kmeans_clusterer.fit_predict()
        
        labels = kmeans_clusterer._cluster_labels
        membership = kmeans_clusterer._membership
        
        return labels, membership.T
    
    def predict_with_softkmeans_v2(self, X):
        """Perform spectral clustering on score matrix X.
        Args:
            X: numpy array of shape (n_samples, n_samples)
        Returns:
            labels: numpy array of shape (n_samples,)
        Raises:
            TypeError: if X has wrong type
            ValueError: if X has wrong shape
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        #  Compute affinity matrix.
        # affinity = utils.compute_affinity_matrix(X)
        affinity = X.copy()
        # Refinement opertions on the affinity matrix.
        for refinement_name in self.refinement_sequence:
            op = self._get_refinement_operator(refinement_name)
            affinity = op.refine(affinity)

        # Perform eigen decomposion.
        (eigenvalues, eigenvectors) = utils.compute_sorted_eigenvectors(
            affinity)
        # Get number of clusters.
        stop_eigenvalue = self.stop_eigenvalue
        # bp()
        k = utils.compute_number_of_clusters(
            eigenvalues, self.max_clusters, stop_eigenvalue)
        if self.min_clusters is not None:
            k = max(k, self.min_clusters)

        # Get spectral embeddings.
        spectral_embeddings = eigenvectors[:, :k]

        # Run K-Means
        # Using custom_dist a custom distance measure can be used
        # Setting custom_dist=cosine is consistent with the paper
        # For custom_dist=None KMeans++ algorithm is used
     
            
        initkmeans_clusterer = KMeans(
                n_clusters=k,
                init="k-means++",
                max_iter=30,
                random_state=0)
            
        initkmeans_clusterer.fit(spectral_embeddings)
        init = initkmeans_clusterer.cluster_centers_
        if k > 1:
            initbeta = 1.0/np.diag(np.cov(spectral_embeddings.T)).mean()
        else:
            initbeta = 1.0/ np.var(spectral_embeddings)
        # bp()
        kmeans_clusterer = custom_distance_kmeans.softKMeans_v2(spectral_embeddings,
                c=k,
                maxiter=10,
                beta=initbeta,
                init=init,
                metric=self.custom_dist)
        kmeans_clusterer.fit_predict()
        
        labels = kmeans_clusterer._cluster_labels
        membership = kmeans_clusterer._membership
        # bp()
        return labels, membership.T