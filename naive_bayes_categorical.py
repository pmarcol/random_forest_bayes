import numpy as np

class NBCategorical(object):

    def __init__(self, smoothing_parameter=0.0):
        self.prob_y = {}
        self.counts_y = {}
        self.smoothing_parameter = smoothing_parameter
    
    def fit(self, X, y):
        if len(X.shape)==1:
            X = X.reshape(-1,1)
        X = X.astype('<U22')
        y = y.astype('<U22')
        
        self.unique_y, counts_y = np.unique(y, return_counts=True)
        self.n_unique_xis = [len(np.unique(X[:,i])) for i in range(X.shape[1])]
        self.list_p_x_given_y = [{} for i in range(X.shape[1])]

        # compute probability and counts of each class in y array
        for cl, n in zip(self.unique_y, counts_y):
            self.prob_y[cl] = n/y.shape[0]
            self.counts_y[cl] = n

        """
        for each feature in X, calculate conditional probability of each category given class of y
        it is stored in list of dictionaries in the following form:
        
        [
            {(x1_unique_cat1, unique_label_1): val, (x1_unique_cat1, unique_label_2): val, ... , (x1_unique_cat1, unique_label_1): val, ...},
            {(x2_unique_cat1, unique_label_1): val, (x2_unique_cat1, unique_label_2): val, ... , (x2_unique_cat1, unique_label_1): val, ...},
            ...
        ]
        """
        # iterate over all columns in X:
        for i in range(X.shape[1]):
            
            # build temporary matrix of two column, being X_i and y
            temp = np.hstack((X[:,i].reshape(-1, 1), y.reshape(-1, 1)))

            # get all unique combinations of values of X_i and y and get their counts
            unique_xi_y, counts_xi_y = np.unique(temp, axis=0, return_counts=True)

            # iterate over the unique pairs:
            for j in range(unique_xi_y.shape[0]):
                
                # save respective conditional probability to xi_dict
                self.list_p_x_given_y[i][tuple(unique_xi_y[j])] = \
                    (counts_xi_y[j]+self.smoothing_parameter)/ \
                        (self.counts_y[unique_xi_y[j, 1]] + self.smoothing_parameter*self.n_unique_xis[i])
                
        return

    def predict(self, X):
        
        X = X.astype('<U22')
        if len(X.shape)==1:
            X = X.reshape(-1,1)
        
        # prepare empty array for row-wise products of conditional probabilities
        products_for_all_uy = np.zeros((X.shape[0], len(self.unique_y)))

        # iterate over all unique labels
        for i in range(len(self.unique_y)):
            
            # prepare empty array for conditional probabilities computed with fit method
            probs = np.zeros(X.shape)
            
            # iterate over all features (columns in X matrix)
            for j in range(X.shape[1]):

                # substitute - when given pair will not be found in the computed conditional probabilities
                # i.e. such combination did not occur in the training set
                subst = self.smoothing_parameter / \
                    (self.counts_y[self.unique_y[i]] + self.smoothing_parameter*self.n_unique_xis[j])

                # lambda function - getting the value from list of dictionaries of conditional probabilities
                translate_to_probs = lambda x: self.list_p_x_given_y[j].get((x, self.unique_y[i]), subst)
                                  
                # utilize the lambda function, vectorize is used for the sake of performance
                probs[:,j] = np.vectorize(translate_to_probs)(X[:,j].reshape(-1))
            
            # row-wise products (as in the Naive Bayes for discrete/categorical features)
            products_for_all_uy[:,i] = np.prod(probs, axis=1)

        # search for max product (for which label the product of probabilities was the largest)
        indices = np.argmax(products_for_all_uy, axis=1)
        
        # lambda function to simply translate index of unique label to the label itself
        translate_to_label = lambda x: self.unique_y[x]
        
        # apply the lambda, obtain vector of predicted labels
        out = np.vectorize(translate_to_label)(indices).reshape(-1,1)
        
        return out