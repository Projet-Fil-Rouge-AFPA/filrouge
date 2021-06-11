#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from annotations import get_annotations_video
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

filename_category = "/Users/Alex/Cours_Telecom/INFMDI 780/LIWC/LIWC_Category.csv"
filename_mapping = "/Users/Alex/Cours_Telecom/INFMDI 780/LIWC/LIWC_Mapping.csv"

folder_Code = '/Users/Alex/Cours_Telecom/INFMDI 780/Code/filrouge/'
folder_Data = '/Users/Alex/Cours_Telecom/INFMDI 780/Data/'
folder_transcript = f'{folder_Data}/Texts_test_man/'

filename_annotations ='https://docs.google.com/\
spreadsheets/d/1Rqu1sJiD-ogc4a6R491JTiaYacptOTqh6DKqhwTa8NA/gviz/tq?tqx=out:csv&sheet=Template'
file_sw = 'french_stop_words.txt'


#Retrieve labels
df_annotations = pd.read_csv(filename_annotations,  header=None).drop([0, 1, 2, 3])

df_category = pd.read_csv(filename_category,header=None)
df_mapping = pd.read_csv(filename_mapping,header=None)
df_mapping.set_index(keys=0,inplace=True)

video_names = set(df_annotations[1].values)

dict = {}

for (j,video) in enumerate(video_names):

    for i in range(1,5):
        text_file = f'{video}_{i}.txt'
        video_name = video
        diapo = i+7
        label = get_annotations_video(filename_annotations, video,'max')[2]
        gender = get_annotations_video(filename_annotations, video,'max')[4]    
        gender_bool = 1.0 if gender == 'H' else 0.0
        group  = j

        dict[text_file] = (video_name, diapo,label[i-1],gender_bool, group)

df_labels = pd.DataFrame.from_dict(dict,columns=['video_name','diapo','label','gender','group'],orient='index')

text_files = df_labels.index

#Retrieve transcripts
transcripts = []

os.chdir(folder_transcript)

for i in range(len(text_files)):

    with open(text_files[i],'r') as file:
        transcript = file.read()
        transcripts.append(transcript)

#Creating LIWC_matrix

dict_list=[]

for transcript in transcripts:
    
    dict_LIWC={i:0 for i in range(1,465)}
    
    words = transcript.split()
    
    for word in words:
        
        if word in df_mapping.index.values:
            categories = df_mapping.loc[word].dropna()
            
            for cat in categories:
                dict_LIWC[cat]+=1
                
    dict_list.append(dict_LIWC)
    
LIWC_matrix = pd.DataFrame(dict_list)


#Keeping only most frequent categories (>1%)
#total = LIWC_matrix.sum().sum()
#pc_cat = LIWC_matrix.sum(axis=0)/total
#Keeping categories>1% of occurence
#pc_cat_red = pc_cat[pc_cat>0.01]
#LIWC_matrix = LIWC_matrix[pc_cat_red.index]

#Trimming matrix
#for i in LIWC_matrix.columns:
    #if LIWC_matrix[i].sum()==0:
        #LIWC_matrix.drop(i,axis=1, inplace=True)

#Classifiers
clf_RF = RandomForestClassifier(criterion='entropy',max_features='log2',random_state=42)

#Leave-One-Interviewer-Out cross-validation
LOGO = LeaveOneGroupOut()
groups = df_labels.group.values

#Normalising the data
scaler = StandardScaler()
X_cv = scaler.fit_transform(LIWC_matrix)

y_cv = df_labels['label'].values

score_list_RF = []
score_list_SVC = []

F1_list_RF = []
F1_list_SVC = []

for train_index, test_index in LOGO.split(X_cv,y_cv,groups):
    X_train, X_test = X_cv[train_index], X_cv[test_index]
    y_train, y_test = y_cv[train_index], y_cv[test_index]
    
    #Train fitting
    clf_RF.fit(X_train,y_train)
            
    #Test accuracy and F1
    score_RF = clf_RF.score(X_test,y_test)
    
    score_list_RF.append(score_RF)
    
    y_pred_RF = clf_RF.predict(X_test)
    F1_RF = f1_score(y_test,y_pred_RF,average='weighted')
    
    F1_list_RF.append(F1_RF)

print(f'RF: {round(np.mean(score_list_RF),4)} / {round(np.mean(F1_list_RF),4)}')