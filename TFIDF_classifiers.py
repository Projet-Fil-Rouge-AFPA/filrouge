#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import spacy
from annotations import get_annotations_video
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

folder_Code = '/Users/Alex/Cours_Telecom/INFMDI 780/Code/'
folder_Data = '/Users/Alex/Cours_Telecom/INFMDI 780/Data/'
folder_transcript = f'{folder_Data}/Texts_test_man/'

filename_annotations ='https://docs.google.com/spreadsheets/d/1Rqu1sJiD-ogc4a6R491JTiaYacptOTqh6DKqhwTa8NA/gviz/tq?tqx=out:csv&sheet=Template'
file_sw = 'french_stop_words.txt'


#Retrieve labels
df_annotations = pd.read_csv(filename_annotations,  header=None).drop([0, 1, 2, 3])

video_names = set(df_annotations[1].values)

dict = {}

for video in video_names:

    for i in range(1,5):
        text_file = f'{video}_{i}.txt'
        label = get_annotations_video(filename_annotations, video,'max')[2]
        gender = get_annotations_video(filename_annotations, video,'max')[4]    
        gender_bool = 1.0 if gender == 'H' else 0.0

        dict[text_file] = (label[i-1],gender_bool)

df_labels = pd.DataFrame.from_dict(dict,columns=['Label','Gender'],orient='index')

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


#Multinomial Naive Bayes
clf = MultinomialNB()
clf_RF = RandomForestClassifier()
clf_SVC = SVC()

#Stratified cross-validation
skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)

X_cv = X_tfidf.toarray()

#np_gender = df_labels['Gender'].to_numpy().reshape(-1,1)
#X_cv_gender = np.concatenate((X_cv,np_gender),axis=1)

y_cv = df_labels['Label'].values

scores_list = []
scores_list_RF = []
scores_list_SVC = []

for train_index, test_index in skf.split(X_cv, y_cv):
    X_train, X_test = X_cv[train_index], X_cv[test_index]
    y_train, y_test = y_cv[train_index], y_cv[test_index]
    
    clf.fit(X_train,y_train)
    clf_RF.fit(X_train,y_train)
    clf_SVC.fit(X_train,y_train)
    
    score = clf.score(X_test,y_test)
    score_RF = clf_RF.score(X_test,y_test)
    score_SVC = clf_SVC.score(X_test,y_test)
    
    scores_list.append(score)
    scores_list_RF.append(score_RF)
    scores_list_SVC.append(score_SVC)

print(f'MNB : {np.mean(scores_list)}')
print(f'RF : {np.mean(scores_list_RF)}')
print(f'SVC : {np.mean(scores_list_SVC)}')