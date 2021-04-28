# Implementation of
'''
Gaudette L., Japkowicz N. (2009) 
Evaluation Methods for Ordinal Classification. 
In: Gao Y., Japkowicz N. (eds) Advances in Artificial Intelligence. Canadian AI 2009. 
Lecture Notes in Computer Science, vol 5549. Springer, Berlin, Heidelberg. 
https://doi.org/10.1007/978-3-642-01818-3_25
'''
import numpy as np
import pandas as pd

def multi_to_binary(y):
    """
    Transform the multi-label target (3 levels of stress) into 2 series of binary targets
    Args:
        y: serie of (int) labels of the training
    Returns:
        y_1: binary targets 0 (0) vs 1,2 (1)
        y_2: binary targets 0,1 (0) vs 2
    """
    y = y.squeeze().astype(int)
    #Convert to binary targets
    y_1 = y.replace(2, 1)
    y_2 = y.replace(1, 0)
    
    targets = [y_1, y_2]
    return targets

def predict_classifiers(list_models, x_test):
    """
    Predict using a list of trained classifiers
    Args:
        list_models: list of trained classifiers
        x_test: dataframe
    Returns:
        preds_argmax: array of prediction
    """
    #Predict
    list_pred = []
    for i in range(len(list_models)):
        y_proba = list_models[i].predict_proba(x_test)[:,1]
        list_pred.append(y_proba)

    #Dic of preds
    preds = {}
    preds[0] = 1 - list_pred[0]
    preds[1] = list_pred[0] - list_pred[1]
    preds[2] = list_pred[1]

    #Get argmax of dataframe -> get the index of the highest probability which corresponds to the label
    preds = pd.DataFrame.from_dict(preds,orient='index').transpose()
    preds_argmax = preds.idxmax(axis="columns")

    return np.array(preds_argmax)

def ordinal_classification(list_models, x_train, y_train, x_test):
    '''
    Use all functions to return an array of prediction
    Args:
        list_models: list of 2 classifiers
        x_train, y_train, x_test: dataframes
    Returns:
        preds_argmax: array of prediction
    """
    '''
    y_train_targets = multi_to_binary(y_train)
    list_models[0].fit(x_train, y_train_targets[0])
    list_models[1].fit(x_train, y_train_targets[1])
    preds_argmax = predict_classifiers(list_models,x_test)
    return preds_argmax