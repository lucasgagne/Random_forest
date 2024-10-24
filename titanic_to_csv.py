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
    df.to_csv('titanic_predictions.csv', index_label='Id')


final_data = pd.read_csv('imputed_testing_titanic_data.csv')
train_data = pd.read_csv('imputed_training_titanic_data.csv')

print(train_data.columns)
training_labels = train_data['survived']
train_data = train_data.drop(columns=['survived'])

my_tree = trees.DecisionTree(is_random=False)
my_tree.fit(train_data, training_labels)
predictions = my_tree.predict(final_data)
print(len(final_data))
print(len(predictions))
print(predictions)


results_to_csv(predictions)