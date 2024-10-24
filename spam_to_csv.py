# A code snippet to help you save your results into a kaggle accepted csv
import pandas as pd
import numpy as np
from sklearn import svm
import decision_tree_starter as trees

# Usage: results_to_csv(clf.predict(X_test))
def results_to_csv(y_test):
    
    y_test = y_test
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df['Category'] = df['Category'].astype(int)
    df.to_csv('spam_predictions.csv', index_label='Id')


train_data = pd.read_csv('processed_spam_training.csv')
final_data = pd.read_csv('processed_spam_testing.csv')

print(train_data.columns)
training_labels = train_data['spam/ham']
train_data = train_data.drop(columns=['spam/ham'])

my_tree = trees.CustomForest(n = 10)
my_tree.fit(train_data, training_labels)
predictions = my_tree.predict(final_data)
print(len(final_data))
print(len(predictions))
print(predictions)


results_to_csv(predictions)