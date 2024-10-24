import scipy.io
import pandas as pd
import numpy as np
mat = scipy.io.loadmat('datasets/spam_data/spam_data.mat')
spam_training_labels = mat['training_labels']
spam_training_data = mat['training_data']
spam_test_data = mat['test_data']

'''Below I will drop Nan values for titanic training data, and use k-neighbors for the test data.'''
'''So far k = 7 for nearest neighbors has performed the best'''
titanic_training = pd.read_csv('datasets/titanic/titanic_training.csv') 
titanic_training_labels = titanic_training['survived']
# print(len(titanic_training.iloc[0]))
titanic_training_data = titanic_training.drop(columns = 'survived')
# print(titanic_training_labels)
# print(len(titanic_training_data.iloc[0]))
# print(len(titanic_training))
# titanic_training = titanic_training.dropna()
# print(len(titanic_training))

titanic_testing = pd.read_csv('datasets/titanic/titanic_testing_data.csv')
# print(titanic_testing)

def get_dist(row1, row2, data):
    #TODO we must decide how to deal with distances to missing values, we can skip columns with missing vals, or we can set them to median
    # row1 = data.loc[row1]
    # row2 = data.loc[row2]
    total_dist = 0
    for col in data: #TODO can change weights and importance of distance for columns.......
        if not pd.isna(data.at[row1, col]) and not pd.isna(data.at[row2, col]): #TODO for now I am just not considering Null vals
            # Get the dtype of the value at the specified row and column
    
            # if data.at[row1, col].dtype == 'int64':  # Assuming you're working with int64 dtype
            if isinstance(data.at[row1, col], int):
                total_dist += (data.loc[row1, col] - data.loc[row2, col])**2
            else:
                if data.at[row1, col] == data.at[row2, col]:
                    # print("the same non int val yooooo")
                    total_dist += 0
                else:
                    total_dist += 5 #TODO this is an arbitrary weight
            
    # print("GOT HERE WHAAAAA")
    return total_dist**0.5

def get_k_nearest(data, col, row, k):
    #return the k nearest neighbors of a point. 
    points = [] #keep array of tuples, where first val is row, second val is dist. We will sort by nearest dist then choose k nearest
    for idx in data.index:
        if not pd.isna(data.at[idx, col]):
            dist = get_dist(row, idx, data)
            points.append((idx, dist))
    points.sort(key = lambda x: x[1])
    k_nearest = points[:k] #get the k nearest
    k_nearest = [pt[0] for pt in k_nearest] #drop the dists we dont need them
    k_nearest = [data.at[idx, col] for idx in k_nearest]  #get actual vals of k_nearest

    return k_nearest

#Go thru each point and each col, get the k nearest for missing vals, and impute as needed
def impute_values(data):
    new_table = data.copy()
    for row in data.index:
        for col in data:
            if pd.isna(data.at[row, col]): #imputation required
                # print("GOT HERE WOOOOOO")
                k_nearest = get_k_nearest(data, col, row, 7)
                #DEPENDING ON THE COLUMN WE DO SOMETHING
                if isinstance(k_nearest[0], int): #if working with ints return median
                    val = np.median(k_nearest)#TODO PLACEHOLDER DELETE THIS
                else: #return most common 
                    val = max(set(k_nearest), key = k_nearest.count)
                    
                
                new_table.at[row, col] = val
    return new_table

def process_pclass(X):
    #map each val in pclass to a str so we treat it categorically in our decision tree (tree will treat strs categorically and everything else as quantitative)
    X = X.copy()
    X['pclass'] = X['pclass'].apply(str)
    return X

def process_ticket(X):
    #get rid of any leading letters/ info my splitting then only keeping the second part (the num), then map int() onto it
    #we are assuming ticket may be useful if it is a continous int, which it may not be, in which case we will simply drop it...
    X = X.copy()
    X['ticket'] = X['ticket'].apply(lambda x: int(x.split()[-1]))
    return X 

def process_cabin(X):
    X = X.copy()
    X['cabin'] = X['cabin'].apply(lambda x: x.split()[0])
    return X


# for row in titanic_training:
#     print("row: ", row)
# print(titanic_training.index)
# print(titanic_training.loc[1, 'survived'])
# print(titanic_training)
print("len of training before: ", len(titanic_training))
print("len after dropping nans: ", len(titanic_training.dropna()))

new_titanic_training = impute_values(titanic_training)
print("new len: ", len(new_titanic_training))
print("len dropping nans: ", len(new_titanic_training.dropna()))
print(new_titanic_training)


# for val in new_titanic_training[]
new_titanic_training.to_csv('imputed_training_titanic_data.csv')

new_titanic_testing = impute_values(titanic_testing)
# new_titanic_testing.to_csv('imputed_testing_titanic_data.csv')
