import scipy.io
import pandas as pd
import numpy as np
mat = scipy.io.loadmat('datasets/spam_data/spam_data.mat')
print(mat.keys())
spam_training_labels = mat['training_labels']
spam_training_data = mat['training_data']
spam_test_data = mat['test_data']
for key in mat.keys():
    print(key, mat[key])
    
features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]

# print(spam_training_data)

print(len(spam_training_data[0]) == len(features))


spam_training_data= pd.DataFrame(spam_training_data, columns=features)
spam_training_labels = pd.DataFrame(spam_training_labels[0], columns = ['spam/ham'])
spam_test_ftrs = pd.DataFrame(spam_test_data, columns = features)





spam_training_data['spam/ham'] =  spam_training_labels['spam/ham']
print(spam_training_data)
spam_training_data.to_csv('processed_spam_training.csv')
spam_test_ftrs.to_csv('processed_spam_testing.csv')