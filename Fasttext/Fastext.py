#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import fasttext.util
import os
import pandas as pd
import numpy as np
import spacy
from annotations import get_annotations_video

fasttext.util.download_model('fr', if_exists='ignore')
ft = fasttext.load_model('./Fasttext/cc.fr.300.bin')

folder_Code = '/Users/Alex/Cours_Telecom/INFMDI 780/Code/filrouge/'
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
    
#Creating word_embedding matrix
rows_doc_nb = len(transcript_lem_list)
columns_nb = 300

matrix_doc = np.ones((rows_doc_nb,columns_nb))

for (i,transcript) in enumerate(transcript_lem_list):
    
    rows_trans_nb = len(transcript.split())
    matrix_trans = np.ones((rows_trans_nb,columns_nb))

    for (j,word) in enumerate(transcript.split()):
        matrix_trans[j,] = ft.get_word_vector(word)
    
    vect_average = matrix_trans.mean(axis=0)
    matrix_doc[i,] = vect_average 
     
#Loading French stop words
#os.chdir(folder_Code)

#input_file = open(file_sw)

#sw_list = []

#for word in input_file:
    #sw_list.append(word[:-1])


