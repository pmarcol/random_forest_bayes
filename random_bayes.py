import numpy as np
from naive_bayes_categorical import NBCategorical
from scipy.stats import mode

class RandomBayes(object):
    
    def __init__(self, n_submodels=10, n_features = None, smooth=0.0, bootstrap=False):
        self.classifiers = [NBCategorical(smoothing_parameter=smooth) for i in range(n_submodels)]
        self.n_submodels = n_submodels
        self.indices_for_classifiers = []
        self.X_features = 0
        self.n_features_for_submodels = n_features
        self.boot = bootstrap

    def fit(self, X, y):
        if len(X.shape)==1:
            X = X.reshape(-1,1)

        # get number of features and number of observations
        self.n_observations, self.X_features = X.shape
        
        # calculate number of features to feed each of the submodels; default = floor(sqrt(X_features))
        if not self.n_features_for_submodels:
            self.n_features_for_submodels = int(np.sqrt(self.X_features))
        
        # randomize feature sets for submodels - save the subsets in form of X's columns' indices
        # fit classifiers
        for i in range(self.n_submodels):
            inds = np.random.choice(self.X_features, self.n_features_for_submodels, False)
            self.indices_for_classifiers.append(inds)
            if self.boot:
                obs = np.random.choice(self.n_observations,self.n_observations,True)
                self.classifiers[i].fit(X[obs[:,None],inds], y)
            else:
                self.classifiers[i].fit(X[:,inds], y)

        return

    def predict(self, X):
        if len(X.shape)==1:
            X = X.reshape(-1,1)
        
        # prepary empty array for votes
        votes = np.zeros((X.shape[0], self.n_submodels), dtype=X.dtype)
        for i in range(self.n_submodels):
            votes[:,i] = (self.classifiers[i].predict(X[:,self.indices_for_classifiers[i]])).reshape(-1)
        
        votes = votes.astype('<U22')
        # the mode of votes will be our prediction
        out = mode(votes, axis=1)[0]
        return out