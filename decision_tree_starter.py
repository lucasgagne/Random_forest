"""
Have Fun!
- 189 Course Staff
"""
import numbers
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from pydot import graph_from_dot_data
import io
from scipy.stats import mode

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number


class DecisionTree:

    def __init__(self, max_depth=None, feature_labels=None, print_decisions = False, decision_point = 0.9, is_random = True):
        self.print_decisions = print_decisions #if this is true, we will print every choice in the predict function.
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes #MYNOTE what is self.data?
        self.decision_point = decision_point #Proportion of classes at which we make a leaf node
        self.is_random = is_random#minimum number of columns to choose to add randomness, can be total num cols

    @staticmethod
    def entropy(y):
        # TODO
        class_counts = {}
        for y_i in y:
            if y_i not in class_counts: class_counts[y_i] = 1
            else: class_counts[y_i] += 1
        total_entropy = 0
        for y_i in y:
            prob_y_i = class_counts[y_i] / len(y)
            total_entropy += -prob_y_i*np.log2(prob_y_i)
            
        return total_entropy

    @staticmethod
    def information_gain(X, y, thresh): 
        # TODO we calculate weighted Hafter (entropy left + entropy right) then subtract it from entropy curr node
        #MYNOTE I am passing in a column for X, not a matrix..
        curr_entropy = DecisionTree.entropy(y)
        thresh, thresh_type = thresh[0], thresh[1]
        if thresh_type == "quant":
            idx0 = np.where(X < thresh)[0]
            idx1 = np.where(X >= thresh)[0]
        elif thresh_type == "qual":
            idx0 = np.where(X == thresh)[0]
            idx1 = np.where(X != thresh)[0]
        else:
            assert 5 == 4, "Invalid thresh type"
        # print("TEST equality: ", len(y) == len(X))
        # print("y; ", y)
        # print("idx 0: ", idx0)
        # print("y: ", y)
        Y_left = y.iloc[idx0]
        Y_right = y.iloc[idx1]
        
        weighted_entropy = (len(Y_left)*(DecisionTree.entropy(Y_left)) + len(Y_right)*(DecisionTree.entropy(Y_right))) / (len(Y_left) + len(Y_right))
        
        return curr_entropy - weighted_entropy

    @staticmethod
    def gini_impurity(X, y, thresh):
        # TODO
        pass

    @staticmethod
    def gini_purification(X, y, thresh):
        # TODO
        pass


    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y.iloc[idx0], y.iloc[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        thresh, thresh_type = thresh
        if thresh_type == "quant":
            idx0 = np.where(X.loc[:, idx] < thresh)[0] #go left on <
            idx1 = np.where(X.loc[:, idx] >= thresh)[0] #go right when >=
        elif thresh_type == "qual":
            # print("THRESH: ", thresh)
            # print("col: ", X.loc[:, idx])
            idx0 = np.where(X.loc[:, idx] == thresh)[0] #Go left on equal
            idx1 = np.where(X.loc[:, idx] != thresh)[0] #go right on not equal
        else: 
            assert 6 == 9, "invalid thresh type"
            
        X0, X1 = X.iloc[idx0], X.iloc[idx1]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        # TODO
        #TODO might want to cahnge base case... base case: if all y are the same, declare this node a leaf, and return.. else recursion
        # count = set()
        # for i in y: count.add(i)
        if self.max_depth != None and self.max_depth == 0: #we will subtract max depth for each additional node... if we get to zero, we must create a leaf, thus we will assign the mode of our labels
            self.pred = mode(y)[0] #pick most common data pt
            
        
        # elif y.nunique() == 1: #base case reached
        elif y.to_list().count(mode(y)[0]) > self.decision_point*len(y):
            # print("BASE CASE: ")
            # print(y)
            self.pred = y.iloc[0] #we are a leaf node and we predict y at this leaf TODO fix this base case, we dont need to get down all to one data type
        
        elif len(y) == 0:
            # print("GOT HERE???")
            self.pred = 1.0
        
        else:#ELSE recursion
            max_info = float("-inf") #track our maximum info gain across splits... 
            max_split_params = (None, None) #track which split resulted in a maximum, (j, thresh), note thresh is itself a tuple.
            max_split_data = (None, None) #track the left and right data where data is a tuple (X, Y)
            #try splits
            # print("X: ", X)
            # print("Y: ", y)
            # for _ in range(9): #TODO fix this, go thru each col/feature and get best split
            col_lbls = X.columns
            # print("COL_LBLS: ", col_lbls)
            # col_lbls = col_lbls[:len(col_lbls)//2]
            # print("NEW COL_LBLS: ", col_lbls)
            z = random.randint(2, len(col_lbls))  # Change this to the number of random columns you want to select
            col_lbls = list(col_lbls)
            if self.is_random:
                col_lbls = random.sample(col_lbls, z)
            # print("NEW: ", random_columns)
            
            for col in col_lbls:
                # i = random.randint(0, len(X.columns)-1)
                # arr = [col for col in X.columns]
                # print("COLS: ", X.columns)
                # print("COLS LOOP: ", [col for col in X.columns])
                # print("X.columns[i]: ", type(X.columns[i]))
                # col = arr[i]
                # print("test: ", X[col])
                #CHOOSE POTENTIAL THRESHOLD DEPENDING ON DATA TYPE. CHECK INT OR NOT INT
                potential_thresholds = self.get_col_thresh(X, col)
                for thresh in potential_thresholds: #TODO threshold is now a tuple as we need to note when the threshold is for quantitative/ categorical data. Must assume thresh always a tuple.. Check every use of thresh if necessary
                    info_gain = DecisionTree.information_gain(X[col], y, thresh=thresh)
                    if info_gain > max_info:
                        max_info = info_gain
                        max_split_params = (col, thresh)
                        X0, Y0, X1, Y1 = self.split(X, y, col, thresh = thresh)
                        max_split_data = ((X0, Y0), (X1, Y1))
            
            #set our node variables, fit out left and right nodes recursively
            # print("[max_split_params[0]: ", max_split_params[0])
            # print("max_split_params[1]: ", max_split_params[1])
            # print("[max_split_data[0][0]: ", max_split_data[0][0])
            # print("[max_split_data[0][1]: ", max_split_data[0][1])
            # print("max_split_data[1]: ", max_split_data[1])
            
            self.split_idx, self.thresh = max_split_params
            # print("BEFORE: ", self.left)
            if self.max_depth != None:
                self.left = DecisionTree(max_depth = self.max_depth - 1, print_decisions=self.print_decisions)
            else:
                self.left = DecisionTree(print_decisions=self.print_decisions)
            # if len(max_split_data[0][0]) == 0:print("GOT HERE WTF")
            # if len(max_split_data[0][1]) == 0:print("GOT HERE WTF 2")
            # if len(max_split_data[1][0]) == 0:print("GOT HERE WTF 3")
            # if len(max_split_data[1][1]) == 0:
            #     print("GOT HERE WTF 4")
            #     print(self.split_idx)
            #     print(self.thresh)
            #     print(X)
            #     print(y)
            #     print("Done with eval")
            self.left.fit(max_split_data[0][0], max_split_data[0][1])
            # print("AFTER: ", self.left)
            if self.max_depth != None:
                self.right = DecisionTree(max_depth=self.max_depth-1, print_decisions=self.print_decisions)
            else:
                self.right = DecisionTree(print_decisions=self.print_decisions)
            self.right.fit(max_split_data[1][0], max_split_data[1][1])
            self.pred = None
            # print("CHECK THIS IS TRUE: ", self.right != None, self.left != None, self.pred == None)
            # print("self.pred: ", self.pred)
    def get_col_thresh(self, X, col):
        #take an array, determine if continous, discrete, quantitative, etc... Return tuple with thresh val and type (to indicate type)
        thresh_vals = []
        # print("X[col]: ", X[col])
        # print("X[col][0]", X[col][0])
 
        if isinstance(X[col].iloc[0], numbers.Number) or isinstance(X[col].iloc[0], np.number): #if int or float assume quantitative data type
            #if quantitative data type, try min, max, 10 percentile, 25 percentile, 50 percentile, 75 percentile, 90 percentile, max as thresh
            percentiles = [5, 10, 25, 30, 50, 70, 90, 95]
            thresh_vals += [(np.percentile(X[col], percent), "quant") for percent in percentiles]
            thresh_vals += [(max(X[col]), "quant"), (min(X[col]), "quant")]

        elif isinstance(X[col].iloc[0], str): #categorical data type
            #if categorical data type, let our potential thresh be every feature, i.e. try all splits and see what maximizes info gain.
            vals = X[col].unique()
            thresh_vals += [(item, "qual") for item in vals]
        else:
            print("THE DATA TYPE: ", type(X[col].iloc[0]))
            assert type(X[col].iloc[0]) == str , "Invalid data type in get_col_thresh"
            
        return thresh_vals

    def predict(self, X):
        # TODO
        #for each point in X, go thru tree recursively
        def recurs(point, node):
            if self.print_decisions:
                print("NEW NODE")
                print(" ")
            if node.pred != None: #we are at a root node, classify the data
                if self.print_decisions: print("so we make prediction: ", node.pred)
                return node.pred
            else:
                #go left or right depending on the thing
                if self.print_decisions:
                    print("THRESHOLD: ", node.thresh)
                    print("split idx: ", node.split_idx)
                    print("val at point: ", point[node.split_idx])
                    print(" ")
                thresh_val, thresh_type = node.thresh
                if thresh_type == 'quant':
                    if point[node.split_idx] >= thresh_val: #TODO  NOTE WE NEED TO MAKE SURE WE ARE ALWAYS DOING RIGHT ON >= AND LEFT ON < OR VICE VERSA!!!!!!!
                        # print("going right")
                        if self.print_decisions: print("So we go right")
                        return recurs(point, node.right)
                    else:
                        # print("going left")
                        if self.print_decisions: print("So we go left")
                        return recurs(point, node.left)
                elif thresh_type == 'qual':
                    if point[node.split_idx] != thresh_val: #TODO  NOTE WE NEED TO MAKE SURE WE ARE ALWAYS DOING RIGHT ON != AND LEFT ON == OR VICE VERSA!!!!!!!
                        # print("going right")
                        if self.print_decisions: print("So we go right")
                        return recurs(point, node.right)
                    else:
                        # print("going left")
                        if self.print_decisions: print("So we go left")
                        return recurs(point, node.left)
                else:
                    assert 5 == 6, "invalid thresh type check for errors."
    
        
        Y = []
        for i in range(len(X)):
            classification = recurs(X.iloc[i, :], self)
            Y.append(classification)
        return Y

    # def __repr__(self):
    #     if self.max_depth == 0:
    #         return "%s (%s)" % (self.pred, self.labels.size)
    #     else:
    #         return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
    #                                        self.thresh, self.left.__repr__(),
    #                                        self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=30):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTree(max_depth=i, feature_labels = self.params) #TODO double check this, we are changing the max depth for each tree, but they are the same

            # DecisionTree(max_depth=i, **self.params) #TODO double check this, we are changing the max depth for each tree, but they are the same
            for i in range(self.n) 
        ]
        self.decision_trees.append(DecisionTree(feature_labels = self.params))
        # self.decision_trees += [DecisionTree(max_depth= i*10 + n) for i in range(10)]
        self.decision_trees += [DecisionTree(max_depth= i*10+ n) for i in range(10)]

        self.decision_trees += [DecisionTree(decision_point=0.9),
                               DecisionTree(decision_point=0.8),
                               DecisionTree(decision_point=0.9),
                               DecisionTree(decision_point=1.0),
                               DecisionTree(decision_point=0.85)]

    def fit(self, X, y):
        # TODO
        i = 0
        for tree in self.decision_trees: #fit all our trees on the data
            indices = range(len(X))
            bootstrapped_idx = np.random.choice(indices, replace = True, size = len(X))
            # print("bootstrapped_idx: ", bootstrapped_idx)
            bootstrapped_X, bootstrapped_y = X.iloc[bootstrapped_idx, :], y.iloc[bootstrapped_idx]        
            tree.fit(bootstrapped_X, bootstrapped_y)
            print("DUN with tree ", i)
            i += 1

    
    def predict(self, X):
        # TODO
        votes = [] #gather all the predictions or votes of our trees, use the most common or mean or median depending on application
        for tree in self.decision_trees:
            vote = tree.predict(X)
            votes.append(vote)
        #Votes is now an array of arrays. For each index in array[0] we need to take the mode.....
        def most_frequent(votes):
            preds = []
            for i in range(len(votes[0])): #for each row in the arrays
                arr = []
                for tree in votes: #go thru each tree and get the mode
                    arr.append(tree[i])
                preds.append(mode(arr)[0])
            return preds
        #get most common vote for this application (binary)
        top_votes = most_frequent(votes)
        return top_votes
        


class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        # params['max_features'] = m #TODO we never use the max_features thing
        self.m = m
        super().__init__(params=params, n=n)
        
class CustomForest(BaggedTrees):
    def __init__(self, params=None, n=8):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTree(max_depth=i, feature_labels = self.params, is_random=False) #TODO double check this, we are changing the max depth for each tree, but they are the same

            # DecisionTree(max_depth=i, **self.params) #TODO double check this, we are changing the max depth for each tree, but they are the same
            for i in range(self.n) 
        ]
        # self.decision_trees.append(DecisionTree(feature_labels = self.params))
        # self.decision_trees += [DecisionTree(max_depth= i*10 + n``) for i in range(10)]
        # self.decision_trees += [DecisionTree(max_depth= i*10+ n) for i in range(20)]

        self.decision_trees += [DecisionTree(decision_point=0.9, max_depth=5),
                               DecisionTree(decision_point=0.8, max_depth=6),
                               DecisionTree(decision_point=0.9, max_depth=7),    
                               DecisionTree(decision_point=1.0),
                               DecisionTree(decision_point=0.85)]
        self.decision_trees += [DecisionTree(decision_point=0.9, max_depth=5, is_random=False),
                               DecisionTree(decision_point=0.8, max_depth=6, is_random=False),
                               DecisionTree(decision_point=0.9, max_depth=7, is_random=False),    
                               DecisionTree(decision_point=1.0, is_random=False),
                               DecisionTree(decision_point=0.85, is_random=False)]
       


class BoostedRandomForest(RandomForest):

    def fit(self, X, y):
        # TODO
        pass
    
    def predict(self, X):
        # TODO
        pass


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack(
        [np.array(data, dtype=float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        # TODO
        pass

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)


# if __name__ == "__main__":
#     dataset = "titanic"
#     # dataset = "spam"
#     params = {
#         "max_depth": 5,
#         # "random_state": 6,
#         "min_samples_leaf": 10,
#     }
#     N = 100

#     if dataset == "titanic":
#         # Load titanic data
#         path_train = 'datasets/titanic/titanic_training.csv'
#         data = genfromtxt(path_train, delimiter=',', dtype=None)
#         path_test = 'datasets/titanic/titanic_testing_data.csv'
#         test_data = genfromtxt(path_test, delimiter=',', dtype=None)
#         y = data[1:, 0]  # label = survived
#         class_names = ["Died", "Survived"]

#         labeled_idx = np.where(y != b'')[0]
#         y = np.array(y[labeled_idx], dtype=float).astype(int)
#         print("\n\nPart (b): preprocessing the titanic dataset")
#         X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
#         X = X[labeled_idx, :]
#         Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
#         assert X.shape[1] == Z.shape[1]
#         features = list(data[0, 1:]) + onehot_features

#     elif dataset == "spam":
#         features = [
#             "pain", "private", "bank", "money", "drug", "spam", "prescription",
#             "creative", "height", "featured", "differ", "width", "other",
#             "energy", "business", "message", "volumes", "revision", "path",
#             "meter", "memo", "planning", "pleased", "record", "out",
#             "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
#             "square_bracket", "ampersand"
#         ]
#         assert len(features) == 32

#         # Load spam data
#         path_train = 'datasets/spam_data/spam_data.mat'
#         data = scipy.io.loadmat(path_train)
#         X = data['training_data']
#         y = np.squeeze(data['training_labels'])
#         Z = data['test_data']
#         class_names = ["Ham", "Spam"]

#     else:
#         raise NotImplementedError("Dataset %s not handled" % dataset)

#     print("Features", features)
#     print("Train/test size", X.shape, Z.shape)
    
#     print("\n\nPart 0: constant classifier")
#     print("Accuracy", 1 - np.sum(y) / y.size)

#     # sklearn decision tree
#     print("\n\nsklearn's decision tree")
#     clf = DecisionTreeClassifier(random_state=0, **params)
#     clf.fit(X, y)
#     evaluate(clf)
#     out = io.StringIO()
#     export_graphviz(
#         clf, out_file=out, feature_names=features, class_names=class_names)
#     # For OSX, may need the following for dot: brew install gprof2dot
#     graph = graph_from_dot_data(out.getvalue())
#     graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)
    
#     # TODO
