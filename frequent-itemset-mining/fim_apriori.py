import os
import numpy as np
import csv
# YOUR CODE HERE (OPTION)
import pandas as pd
from csv import reader
import itertools


def Find_Itemset_Cover(X, list_transactions):
    """
    Function find the set of transaction containing X.
    
    * Parameter:
    X -- a 1D python list, indicate an itemset. For example ['milk', 'coffee', ...]
    list_transactions - a 2D python list, indicate the list of transactions
    
    * Return:
    list_transaction_cover -- a python list, indicate the list of transaction_id containing X. 
                                For example: [1, 2, 100, ...]
    """
    list_transaction_cover = None
    

    list_transaction_cover = [] 
    for index, trans in enumerate(list_transactions): 
        in_set = True
        for item in X: 
            if item not in trans: 
                in_set = False
                break
        if in_set:
            list_transaction_cover.append(index)

    
    return list_transaction_cover


def Cal_Absolute_Support(X, list_transactions):
    """
    Function calculate absolute support of itemset X.
    
    * Parameter:
    X -- a 1D python list, indicate an itemset. For example ['milk', 'coffee', ...]
    list_transactions - a 2D python list, indicate the list of transactions
    
    * Return:
    abs_support -- an integer, indicate the number of transactions containing X
    """
    
    abs_support = None
    
    list_transaction_cover = Find_Itemset_Cover(X, list_transactions)
    abs_support = len(list_transaction_cover)
    
    
    return abs_support


def Cal_Relative_Support(X, list_transactions):
    """
    Function calculate absolute support of itemset X.
    
    * Parameter:
    X -- a 1D python list, indicate an itemset. For example ['milk', 'coffee', ...]
    list_transactions - a 2D python list, indicate the list of transactions
    
    * Return:
    relative_support -- a float, indicate the probability of transactions containing X
    """
    
    relative_support = None

    relative_support = round(Cal_Absolute_Support(X, list_transactions) / len(list_transactions), 4)
    
    return relative_support


def Check_Frequent_Itemset(X, list_transactions, min_sup):
    """
    Function check the itemset X is a frequent of not
    
    * Parameter:
    X -- a 1D python list, indicate an itemset. For example ['milk', 'coffee', ...]
    list_transactions -- a 2D python list, indicate the list of transactions
    min_support -- a float, indicate the minimum support value
    
    * Return:
    is_frequent -- a boolean value. True if X is frequent. False otherwise.
    """
    
    is_frequent = None

    is_frequent = Cal_Relative_Support(X, list_transactions) >= min_sup
    
    return is_frequent

def Cal_Self_Join(itemset_1, itemset_2):
    """
    Function perform self-joining step between itemset_1 and itemset_2
    
    * Parameter:
    itemset_1 -- a 1D python list, indicate the itemset. For example ['candy', 'coffee', 'fruit']
    itemset_2 -- a 1D python list, indicate the itemset. For example ['candy', 'coffee', 'milk']
    
    * Return:
    join_itemset -- a 1D python list, indicate the join itemset. For example ['candy', 'coffee', 'fruit', 'milk'].
                        In case itemset_1 and itemset_2 can not join. Return []
    """
    
    join_itemset = None
    prefix_sas = True


    for i in range(len(itemset_1) - 1):
        if itemset_1[i] != itemset_2[i]:
            prefix_sas = False
            break
    

    if len(itemset_1) == len(itemset_2) and prefix_sas and itemset_1[-1] != itemset_2[-1]:
        join_itemset = list(set(itemset_1 + itemset_2)) 
    else:
        join_itemset = []

    
    return sorted(join_itemset)


def findsubsets(s, n):
    return list(itertools.combinations(s, n))

def Prune_By_Apriori(list_candidate, list_previous_frequent_itemset):
    """
    Function prune by Apriori 
    
    * Parameter:
    list_candidate -- a 2D python list, indicate the C_(k+1)
    list_previous_frequent_itemset -- a 2D python list, indicate the L_k
    
    * Return:
    new_list_candidate -- a 2D python list, indicate the C_(k+1) after prune redundant candidate.
    """
    
    new_list_candidate = None
    new_list_candidate = []


    for candidate in list_candidate:
        list_frequent_itemset = [cans for cans in findsubsets(candidate, len(candidate) - 1)]
        check = True
        for i in findsubsets(candidate, len(candidate) - 1):
            if list(i) not in list_previous_frequent_itemset:
                check = False
        if check:
            new_list_candidate.append(candidate)
        
    
    return new_list_candidate


def Frequent_Itemset_Mining_Apriori(list_transactions, list_items, min_sup):
    """
    Function perform  frequent itemset mining by Apriori algorithm
    
    * Parameter:
    list_transactions -- a 2D python list, indicate the list of transactions
    list_items -- a 1D python list, indicate all item in database. For example: [coffee, candy, ...]
    min_sup -- a float, indicate the minimum support value

    * Return:
    list_frequent_itemset -- a 2D python list, indicate the list of frequent itemset
    """
    
    list_frequent_itemset = None
    
    list_frequent_itemset = []
    prev = [] 

    # intitialize the previous 
    for i in range(len(list_items)):
        if Check_Frequent_Itemset([list_items[i]], list_transactions, min_sup):
            list_frequent_itemset.append([list_items[i]])
            prev.append([list_items[i]])

    while True: 
        candidates = [] 
        for i in range(len(prev) - 1):
            for j in range(i + 1, len(prev)):
                candidate = Cal_Self_Join(prev[i], prev[j])
                if not (candidate == []) and (candidate not in candidates):
                    candidates.append(candidate)

        pruned_candidates = Prune_By_Apriori(candidates, prev)

        true_freq_candidates = []
        for candidate in pruned_candidates:
            if Check_Frequent_Itemset(candidate, list_transactions, min_sup):
                true_freq_candidates.append(candidate)
        prev = true_freq_candidates

        if len(prev) <= 0:
            break
        list_frequent_itemset += prev

    return list_frequent_itemset

if __name__ == '__main__':
    list_transactions = None

    list_transactions = []
    with open('groceries.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            list_transactions.append(sorted(row))
    list_transactions


    list_items = None

    list_items = set([])
    for trans in list_transactions:
        for item in trans:
            list_items.add(item)
    list_items = sorted(list(list_items))


    min_support = 0.1

    list_fim = Frequent_Itemset_Mining_Apriori(list_transactions, list_items, min_support)

    print(list_fim)