import numpy as np
import decision_tree_starter as Trees
import pandas as pd

spam_data = pd.read_csv('processed_spam_training.csv')
spam_data_labels = spam_data['spam/ham']
spam_data_features = spam_data.drop(columns = 'spam/ham')
# print("NUM COLS: ", len(spam_data_features.columns))
spam_data_features = spam_data_features.iloc[:, :10]
# print("SPAM: ", spam_data_features)
# print("spam lbls: ", spam_data_labels)

titanic_data = pd.read_csv('imputed_training_titanic_data.csv')
titanic_data_features = titanic_data.iloc[:, 2:]
titanic_data_labels = titanic_data['survived']
titanic_data_features = titanic_data_features.drop(columns=["cabin"])
# print("FTRS: ")
# print(titanic_data_features)
# print("LABELS: ")
# print(titanic_data_labels)
# print(titanic_data_labels.nunique())
# print(len(titanic_data))
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

titanic_val_labels, titanic_val_ftrs, titanic_train_labels, titanic_train_ftrs = validation_split(titanic_data_features, titanic_data_labels)
spam_val_labels, spam_val_ftrs, spam_train_labels, spam_train_ftrs = validation_split(spam_data_features, spam_data_labels)

def eval_tree_performance(val_ftrs, train_ftrs, val_lbls, train_lbls):
    #get predictions
    tree = Trees.DecisionTree(decision_point=0.8, max_depth=5, is_random=False)
    tree.fit(train_ftrs, train_lbls)
    predictions = tree.predict(val_ftrs)
    print("predictions: ", predictions)
    #check performance
    score = 0
    for i in range(len(predictions)):
        if predictions[i] == val_lbls.iloc[i]:
            score += 1
    val_acc = score / len(predictions)
    
    predictions = tree.predict(train_ftrs)
    tot = 0
    for i in range(len(predictions)):
        if predictions[i] == predictions[i]:
            tot += 1
    train_acc = tot/len(predictions)
    
    return val_acc, train_acc

def eval_forest_performance(val_ftrs, train_ftrs, val_lbls, train_lbls):
    # my_forest = Trees.RandomForest(n = 30)
    my_forest = Trees.CustomForest(n = 8)
    my_forest.fit(train_ftrs, train_lbls)
    predictions = my_forest.predict(val_ftrs)
    print("PREDICTIONS forest: ", predictions)
    tot = 0
    incorrect = []
    for i in range(len(predictions)):
        if predictions[i] == val_lbls.iloc[i]:
            tot += 1 
        else:
            incorrect.append(i)
    val_acc = tot / len(predictions)
    
    predictions = my_forest.predict(train_ftrs)
    tot = 0
    for i in range(len(predictions)):
        if predictions[i] == predictions[i]:
            tot += 1
    train_acc = tot/len(predictions)
    print("Incorrect: ", incorrect)
    return val_acc, train_acc


titanic_tree_acc = eval_tree_performance(titanic_val_ftrs, titanic_train_ftrs, titanic_val_labels, titanic_train_labels)
# titanic_tree_train_acc = eval_tree_performance(titanic_train_ftrs, titanic_train_ftrs, titanic_train_labels, titanic_train_labels)
ttnc_val_acc, ttnc_test_acc = titanic_tree_acc
print("TITANIC TREE VAL ACCURACY: ", ttnc_val_acc)
print("TITANIC TREE TEST ACCURACY: ", ttnc_test_acc)

spam_tree_acc = eval_tree_performance(spam_val_ftrs, spam_train_ftrs, spam_val_labels, spam_train_labels)
spam_tree_val_acc, spam_tree_test_acc = spam_tree_acc
print("SPAM TREE VAL ACC: ", spam_tree_val_acc)
print("SPAM TREE TEST ACC: ", spam_tree_test_acc)

titanic_forest_acc = eval_forest_performance(titanic_val_ftrs, titanic_train_ftrs, titanic_val_labels, titanic_train_labels)
titanic_forest_val_acc, titanic_forest_test_acc = titanic_forest_acc
print("TITANIC FOREST VAL ACCURACY: ", titanic_forest_val_acc)
print("TITANIC FOREST TEST ACCURACY: ", titanic_forest_test_acc)

spam_forest_acc = eval_forest_performance(spam_val_ftrs, spam_train_ftrs, spam_val_labels, spam_train_labels)
spam_forest_val_acc, spam_forest_test_acc = spam_forest_acc
print("SPAM FOREST VAL ACCURACY: ", spam_forest_val_acc)
print("SPAM FOREST TEST ACCURACY: ", spam_forest_test_acc)
