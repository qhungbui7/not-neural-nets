import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
import sklearn.model_selection



def load_data(file_path: str, split: bool = True):
    """
    Load data from file_path and return numpy data

    Parameters
    ----------
    file_path : str
        The path of input data file (tab separated).
    split : bool
        Whether or not to return test set.

    Returns
    ----------
    (X_train, y_train), (None, None): training numpy array if split = False, else
    (X_train, y_train), (X_test, y_test): training and testing numpy array if split = True
    """

    # drop duplicates samples
    data = pd.read_csv(file_path, sep="\t").drop_duplicates().reset_index(drop=True)
    X = data.copy().iloc[:,0:-1]
    y = data.copy().iloc[:,-1] 
    if split: 
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=520)
    else: 
        X_train = X
        y_train = y
        X_test = None
        y_test = None


    return (X_train, y_train), (X_test, y_test)

def entropy_at_node(labels):
    num_can = len(labels)
    list_labels = labels.unique()
    entropy = 0 
    for label in list_labels:
        if len(labels[labels == label]) != 0: 
            entropy += len(labels[labels == label])  / num_can * np.log(len(labels[labels == label])  / num_can)
    return -entropy

def cal_potential(data, labels, chosen_feat):
    list_cat = data[chosen_feat].unique()
    num_sam = len(labels)
    entropy = 0
    label_leaves = set([])

    for cat in list_cat: # analyze each branch
        labels_subset = labels[data[chosen_feat] == cat]
        leng = len(labels_subset)
        if leng != 0:
            entropy += leng / num_sam * entropy_at_node(labels_subset)
            for label in labels_subset.unique(): 
                label_leaves.add(label) 

    return entropy, label_leaves

def choose_feature(data, labels, avail_feats_ori):
    avail_feats = avail_feats_ori.copy()
    current_node_entropy = entropy_at_node(labels)
    best_information_gain = -1 # current_node_entropy
    best_feat = None 
    is_leaf_node = False
    label_if_pure = None

    if len(avail_feats) == 0: # case that there feat are all used

        best_label = None
        best_num = 0
        
        for label in labels.unique():
            if len(data[labels == label]) > best_num:

                best_num = len(data[labels == label])
                best_label = label

        best_feat = None 
        is_leaf_node = True 
        label_if_pure = best_label
    else: # still have feat to use
        for feat in avail_feats:         
            neg_entropy, label_leaves = cal_potential(data, labels, feat)
            ig = current_node_entropy - neg_entropy

            if ig > best_information_gain:
                best_information_gain = ig 
                best_feat = feat 
            
            if neg_entropy == 0 and len(label_leaves) == 1: # already pure case
                best_feat = feat 
                is_leaf_node = True
                label_if_pure = list(label_leaves)[0] 
                break

    return best_feat, is_leaf_node, label_if_pure


class Node:
    def __init__(self):
        self.feat = None
        self.list_child_cat = [] 
        self.list_children = [] 
        self.label = None
        
    def update(self, feat, list_child_cat, is_leaf_node, label=None):
        if not is_leaf_node:
            self.feat = feat
            self.list_child_cat = list_child_cat
            self.list_children = [Node() for i in range(len(list_child_cat))]
        
        else:
            self.label = label

class DecisionTree:
    def __init__(self):
        self.root = Node()

    def fit(self, X_train, y_train):
        avail_feats = list(X_train.columns)
        self.build(X_train, y_train, self.root, avail_feats)
    
    def predict(self, X_test):
        results = [] 
        for index, sample in X_test.iterrows():
            results.append(self.go_deeper(self.root, sample))
        return results
        
    def visualize(self):
        print('Decision Tree structure: ')
        self.visualize_node(self.root, 0, tab='')

    # utilities part
    def build(self, X_train, y_train, node: Node, avail_feats_ori):
        avail_feats = avail_feats_ori.copy()
        feat, is_leaf_node, label_if_leaves = choose_feature(X_train, y_train, avail_feats_ori=avail_feats)

        if not is_leaf_node: 
            avail_feats.remove(feat)
            node.update(feat, X_train[feat].unique(), is_leaf_node, label=None)
            for index in range(len(node.list_children)):
                X_subset = X_train[X_train[feat] == node.list_child_cat[index]]
                y_subset = y_train[X_train[feat] == node.list_child_cat[index]]
                self.build(X_subset, y_subset, node.list_children[index], avail_feats)
        
        else: 
            node.update(None, list_child_cat=[], is_leaf_node=True, label=label_if_leaves)

    def visualize_node(self, node, index, tab):
        if node.label:
            print(f'{tab}Label: {node.label}' )
        else: 
            print(f'{tab}Attribute: {node.feat} (Level {index})' )
            for i, cat in enumerate(node.list_child_cat): 
                print(f'{tab}-Branch: {cat}' )
                self.visualize_node(node.list_children[i], index + 1, tab + '\t')

    def go_deeper(self, node, sample): # recursively go through the tree
        if not node.label:
            for i in range(len(node.list_child_cat)):
                if node.list_child_cat[i] == sample[node.feat]:
                    return self.go_deeper(node.list_children[i], sample)
        else: 
            return node.label

if __name__ == '__main__':
    tree = DecisionTree()

    (X_train, y_train), (X_test, y_test) = load_data("data/titanic2.txt", split=False)

    tree.fit(X_train, y_train)

    y_hat_train = tree.predict(X_train) 
    acc_train = accuracy_score(y_train, y_hat_train)
    y_hat_test = tree.predict(X_test) 
    acc_test = accuracy_score(y_test, y_hat_test)

    tree.visualize()

    print(f'Accuracy of the model for the tennis training set: {acc_train}')
    print(f'Accuracy of the model for the tennis test set: {acc_test}')
