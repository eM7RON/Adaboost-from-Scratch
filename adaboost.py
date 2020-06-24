from sklearn.tree import DecisionTreeClassifier
import numpy as np

class AdaBoost:
    '''
    A simple implementation of the SAMME ada boost for multiclass classification which uses
    sklearn's DecisionTreeClassifier as a weak classifier. I tried to make it be compatible
    with any weak classifier which can be input as argument 'classifier'. Please see sklearn's 
    DecisionTreeClassifier documentation for how some of the attributes/arguments work.
    
    SAMME Ref: http://ww.web.stanford.edu/~hastie/Papers/SII-2-3-A8-Zhu.pdf
    '''
    
    def __init__(
                 self,
                 n_estimators=20,
                 classifier=DecisionTreeClassifier,
                 attr={},
                 random_state=117,
                 ):
        '''
        
        Args:
            n_estimators - int, number of weak classifier instance in the ensemble
            classifier   - class with fit and predict method
            attr         - dict, string: value, attributes of the weak learners
            random_state - hashable, something to seed the random generator
            
        '''
        
        seed(random_state)
        self.n_estimators = n_estimators
        self.classifier = classifier
        self.attributes = attr
    
    def fit(self, X, y):
        
        '''
        Creates a forest of decision trees using a random subset of data and
        features
        
        Args:
            X - array-like, sample training data, shape[n_samples, n_features]
            y - array-like, target labels, shape[n_samples]
        '''

        self.ensemble  = []
        self.n_classes = len(np.unique(y))
        self.alphas    = np.empty(self.n_estimators)
        
        logk      = np.log(self.n_classes - 1.) # log(k - 1)
        n_samples = len(X)
        w = np.ones(n_samples) / n_samples # weights

        for i in range(self.n_estimators):

            # initialize weak predictor 
            clf = self.classifier()

            # set attributes for the weak predictor
            for attr, value in self.attributes.items():
                setattr(clf, attr, value)

            clf.fit(X, y, sample_weight=w)
            predictions = clf.predict(X)
            error = predictions != y

            self.ensemble.append(clf)
            partition = np.array([j if j else -1 for j in error])
            epsilon = sum(w * error) / sum(w) # + regularizer
            
            alpha = np.log((1. - epsilon) / epsilon) + logk
            w *= np.exp(alpha * partition)
            w /= np.sum(w) # normalize so sum(w) == 1
            self.alphas[i] = alpha

    def predict(self, X):
        '''
        Predict the class of each sample in X
    
        Args:
            X           - array-like, sample training data, shape[n_samples, n_features]
        
        Returns:
            predictions - array-like, predicted labels, shape[n_samples]

        '''
        predictions = np.zeros([len(X), self.n_classes])
        for i in range(len(self.ensemble)):
            predictions += self.alphas[i] * self.ensemble[i].predict_proba(X)
        return np.argmax(predictions, axis=1).astype(int)