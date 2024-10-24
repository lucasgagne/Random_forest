import numpy as np
import decision_tree_starter as Trees
import pandas as pd
import matplotlib.pyplot as plt


spam_data = pd.read_csv('processed_spam_training.csv')
spam_data_labels = spam_data['spam/ham']
spam_data_features = spam_data.drop(columns = 'spam/ham')
# print("NUM COLS: ", len(spam_data_features.columns))
spam_data_features = spam_data_features.iloc[:, :10]


def validation_split(features, labels):
    #split features and labels into training and validation splits.. leave 1/5 for validation.
    validation_idx = np.random.choice(range(len(features)), size = len(features)//5, replace=False)#select len(features)/5 vals from range or indices in features
    
    # validation_labels = []
    # validation_features = []
    # train_labels = []
    # train_features = []
    # for i in range(len(features)):
    #     if i in validation_idx:
    #         validation_labels.append(labels.iloc[i])
    #         validation_features.append(features.iloc[i, :])
    #     else:
    #         train_labels.append(labels.iloc[i])
    #         train_features.append(features.iloc[i, :])
    validation_labels = labels.iloc[validation_idx]
    validation_features= features.iloc[validation_idx, :]
    mask = ~features.index.isin(validation_idx)
    train_labels = labels.iloc[mask]
    train_features = features.iloc[mask]
    # print("train len: ", len(features), (len(train_features), len(train_labels)))
    # print("train len: ", len(features), (len(validation_features), len(validation_labels)))
    return validation_labels, validation_features, train_labels, train_features

spam_val_labels, spam_val_ftrs, spam_train_labels, spam_train_ftrs = validation_split(spam_data_features, spam_data_labels)
spam_train_ftrs, spam_train_labels = spam_train_ftrs.iloc[:100, :], spam_train_labels.iloc[:100] #TODO NOTE we are cutting the set short temporarily for testing

'''PART 2, follow the steps of the decision tree in a prediction of a point'''
my_tree = Trees.DecisionTree(print_decisions = True) #adding print decision variable to check our results
my_tree.fit(spam_train_ftrs, spam_train_labels)
ftr_pnt = spam_val_ftrs.iloc[:1, :]
ftr_lbl = spam_val_labels.iloc[:1]
my_tree.predict(ftr_pnt) #WE WILL ADD PRINT STATEMENTS TO OUR PREDICT FUNCTION TO FOLLOW THE DECISIONS IT MADE

'''Validation split was made above in validation_split. Now we will loop thru depths and track accuracies, then plot it '''
def get_accuracy(predictions, labels):
    tot = 0
    for i in range(len(predictions)):
        if predictions[i] == labels.iloc[i]:
            tot += 1
    return tot / len(predictions)


accuracies = []
depths = list(range(1, 41))
for i in range(1, 41):
    test_tree = Trees.DecisionTree(max_depth=i, is_random=False, decision_point=1.0)
    print("Done with tree ", i)
    test_tree.fit(spam_train_ftrs, spam_train_labels)
    predictions = test_tree.predict(spam_val_ftrs)
    accuracy = get_accuracy(predictions, spam_val_labels)
    accuracies.append(accuracy)
    
# Plotting
plt.plot(depths, accuracies, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Depth')

# Displaying the plot
plt.grid(True)
plt.show()


