#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import spacy
from annotations import get_annotations_video
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix

folder_Code = '/Users/Alex/Cours_Telecom/INFMDI 780/Code/filrouge/'
folder_Data = '/Users/Alex/Cours_Telecom/INFMDI 780/Data/'
folder_transcript = f'{folder_Data}/Texts_test_man/'

filename_annotations ='https://docs.google.com/\
spreadsheets/d/1Rqu1sJiD-ogc4a6R491JTiaYacptOTqh6DKqhwTa8NA/gviz/tq?tqx=out:csv&sheet=Template'
file_sw = 'french_stop_words.txt'


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

#Retrieve transcripts
transcripts = []

os.chdir(folder_transcript)

for i in range(len(text_files)):

    with open(text_files[i],'r') as file:
        transcript = file.read()
        transcripts.append(transcript)

#Lemmatization
nlp = spacy.load('fr_core_news_md')

transcript_lem_list = []

for trans in transcripts:
    
    trans_lem = []
    
    doc = nlp(trans)   
    for token in doc:
        trans_lem.append(token.lemma_)
        
    trans_join = " ".join(trans_lem)
    transcript_lem_list.append(trans_join)
    
#Loading French stop words
os.chdir(folder_Code)

input_file = open(file_sw)

sw_list = []

for word in input_file:
    sw_list.append(word[:-1])
    
#Vectorization of the transcripts
vectorizer = CountVectorizer(stop_words=sw_list)

X_vect = vectorizer.fit_transform(transcript_lem_list)

#TF IDF
tfidf = TfidfTransformer()

X_tfidf = tfidf.fit_transform(X_vect)

#Classifiers
clf_MNB = MultinomialNB(alpha=0.8)
clf_RF = RandomForestClassifier(criterion='entropy',max_features='sqrt',random_state=42)
clf_SVC = LinearSVC(C=0.5)

#Leave-One-Interviewer-Out cross-validation
LOGO = LeaveOneGroupOut()
groups = df_labels.Group.values

X_cv = X_tfidf.toarray()

#np_gender = df_labels['Gender'].to_numpy().reshape(-1,1)
#X_cv_gender = np.concatenate((X_cv,np_gender),axis=1)

y_cv = df_labels['Label'].values

score_list_MNB= []
score_list_RF = []
score_list_SVC = []

F1_list_MNB= []
F1_list_RF = []
F1_list_SVC = []

confmat_list_MNB = []

for train_index, test_index in LOGO.split(X_cv,y_cv,groups):
    X_train, X_test = X_cv[train_index], X_cv[test_index]
    y_train, y_test = y_cv[train_index], y_cv[test_index]
    
    #Train fitting
    clf_MNB.fit(X_train,y_train)
    clf_RF.fit(X_train,y_train)
    clf_SVC.fit(X_train,y_train)
            
    #Test accuracy and F1
    score_MNB = clf_MNB.score(X_test,y_test)
    score_RF = clf_RF.score(X_test,y_test)
    score_SVC = clf_SVC.score(X_test,y_test)
    
    score_list_MNB.append(score_MNB)
    score_list_RF.append(score_RF)
    score_list_SVC.append(score_SVC)
    
    y_pred_MNB = clf_MNB.predict(X_test)
    F1_MNB = f1_score(y_test,y_pred_MNB,average='weighted')
    
    y_pred_RF = clf_RF.predict(X_test)
    F1_RF = f1_score(y_test,y_pred_RF,average='weighted')
    
    y_pred_SVC = clf_SVC.predict(X_test)
    F1_SVC = f1_score(y_test,y_pred_SVC,average='weighted')
    
    
    F1_list_MNB.append(F1_MNB)
    F1_list_RF.append(F1_RF)
    F1_list_SVC.append(F1_SVC)
    
    confmat_MNB = confusion_matrix(y_test, y_pred_MNB)
    confmat_list_MNB.append(confmat_MNB)

print(f'MNB: {round(np.mean(score_list_MNB),4)} / {round(np.mean(F1_list_MNB),4)}')
print(f'RF: {round(np.mean(score_list_RF),4)} / {round(np.mean(F1_list_RF),4)}')
print(f'SVC: {round(np.mean(score_list_SVC),4)} / {round(np.mean(F1_list_SVC),4)}')

#print(np.sum(confmat_list_MNB, axis=0))