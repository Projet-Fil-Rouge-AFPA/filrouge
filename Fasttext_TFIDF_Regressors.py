#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from annotations import get_annotations_video
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
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

#Regressors
Ridge = Ridge(alpha=10)
RF = RandomForestRegressor(criterion='mse',max_features='sqrt',random_state=42)
SVR = LinearSVR(C=0.5)

#Leave-One-Interviewer-Out cross-validation
LOGO = LeaveOneGroupOut()
groups = df_labels.Group.values

X_cv = X_vect
y_cv = df_labels['Label'].values

RMSE_list_Rid = []
RMSE_list_RF = []
RMSE_list_SVR = []

for train_index, test_index in LOGO.split(X_cv,y_cv,groups):
    X_train, X_test = X_cv[train_index], X_cv[test_index]
    y_train, y_test = y_cv[train_index], y_cv[test_index]
    
    #Train fitting
    Ridge.fit(X_train,y_train)
    RF.fit(X_train,y_train)
    SVR.fit(X_train,y_train)
            
    #Test RMSE
    y_pred_Rid = Ridge.predict(X_test)
    RMSE_Rid = np.sqrt(np.mean((y_pred_Rid - y_test)**2))
    
    y_pred_RF = RF.predict(X_test)
    RMSE_RF= np.sqrt(np.mean((y_pred_RF - y_test)**2))
    
    y_pred_SVR = SVR.predict(X_test)
    RMSE_SVR = np.sqrt(np.mean((y_pred_SVR - y_test)**2))
    
    RMSE_list_Rid.append(RMSE_Rid)
    RMSE_list_RF.append(RMSE_RF)
    RMSE_list_SVR.append(RMSE_SVR)

print(f'Ridge: {round(np.mean(RMSE_list_Rid),4)}')
print(f'RF: {round(np.mean(RMSE_list_RF),4)}')
print(f'SVR: {round(np.mean(RMSE_list_SVR),4)}')

