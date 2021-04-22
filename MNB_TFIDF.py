#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold

folder_Code = '/Users/Alex/Cours_Telecom/INFMDI 780/Code/'
folder_Data = '/Users/Alex/Cours_Telecom/INFMDI 780/Data/'
folder_transcript = f'{folder_Data}/Texts_test_man/'

file_labels = 'Annotations_man.csv'
file_sw = 'french_stop_words.txt'


#Retrieve labels
filename = f'{folder_Code}{file_labels}'

labels = pd.read_csv(filename, header=None)

files = labels[0].to_list()

text_files = []

for file in files:    
    text_files.append(file)

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

#Stratified cross-validation
skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)

X_cv = X_tfidf.toarray()

y_cv = labels[1].values

scores_list = []

for train_index, test_index in skf.split(X_cv, y_cv):
    X_train, X_test = X_cv[train_index], X_cv[test_index]
    y_train, y_test = y_cv[train_index], y_cv[test_index]
    
    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    scores_list.append(score)

print(np.mean(scores_list))