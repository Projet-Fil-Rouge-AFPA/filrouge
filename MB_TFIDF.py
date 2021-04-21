#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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

X = vectorizer.fit_transform(transcript_lem_list)

#Multinomial Naive Bayes
clf = MultinomialNB()

y = labels[1].values

clf.fit(X.toarray(),y)

score = clf.score(X.toarray(),y)

print(score)