#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from annotations import get_annotations_video
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut

folder_Code = '/Users/Alex/Cours_Telecom/INFMDI 780/Code/filrouge/'
folder_Data = '/Users/Alex/Cours_Telecom/INFMDI 780/Data/'
file_word_embedding = f'{folder_Code}/Fasttext/Matrix_tfidf.npy'

filename_annotations ='https://docs.google.com/\
spreadsheets/d/1Rqu1sJiD-ogc4a6R491JTiaYacptOTqh6DKqhwTa8NA/gviz/tq?tqx=out:csv&sheet=Template'

#Retrieve labels
df_annotations = pd.read_csv(filename_annotations,  header=None).drop([0, 1, 2, 3])

video_names = set(df_annotations[1].values)

dict = {}

for (j,video) in enumerate(video_names):

    for i in range(1,5):
        text_file = f'{video}_{i}.txt'
        label = get_annotations_video(filename_annotations, video,'max')[2]
        gender = get_annotations_video(filename_annotations, video,'max')[4]    
        gender_bool = 1.0 if gender == 'H' else 0.0
        group  = j

        dict[text_file] = (label[i-1],gender_bool, group)

df_labels = pd.DataFrame.from_dict(dict,columns=['Label','Gender','Group'],orient='index')

text_files = df_labels.index

#Retrieve Word Embedding
X_vect = np.load(file_word_embedding)

#Classifiers
clf_RF = RandomForestClassifier(criterion='entropy',max_features='sqrt',random_state=42)
clf_SVC = LinearSVC(C=0.5)

#Leave-One-Interviewer-Out cross-validation
LOGO = LeaveOneGroupOut()
groups = df_labels.Group.values

X_cv = X_vect
y_cv = df_labels['Label'].values

score_list_RF = []
score_list_SVC = []

F1_list_RF = []
F1_list_SVC = []

for train_index, test_index in LOGO.split(X_cv,y_cv,groups):
    X_train, X_test = X_cv[train_index], X_cv[test_index]
    y_train, y_test = y_cv[train_index], y_cv[test_index]
    
    #Train fitting
    clf_RF.fit(X_train,y_train)
    clf_SVC.fit(X_train,y_train)
            
    #Test accuracy and F1
    score_RF = clf_RF.score(X_test,y_test)
    score_SVC = clf_SVC.score(X_test,y_test)
    
    score_list_RF.append(score_RF)
    score_list_SVC.append(score_SVC)
    
    
    y_pred_RF = clf_RF.predict(X_test)
    F1_RF = f1_score(y_test,y_pred_RF,average='weighted')
    
    y_pred_SVC = clf_SVC.predict(X_test)
    F1_SVC = f1_score(y_test,y_pred_SVC,average='weighted')
    
    
    F1_list_RF.append(F1_RF)
    F1_list_SVC.append(F1_SVC)
    
print(f'RF: {round(np.mean(score_list_RF),4)} / {round(np.mean(F1_list_RF),4)}')
print(f'SVC: {round(np.mean(score_list_SVC),4)} / {round(np.mean(F1_list_SVC),4)}')

