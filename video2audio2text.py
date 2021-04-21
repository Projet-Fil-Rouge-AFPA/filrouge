#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import moviepy.editor as mp
import speech_recognition as sr
import pandas as pd

def load_write_audio(rep,video_dir,file):
    
    video_file = mp.VideoFileClip(f'{rep}/{video_dir}/{file}')
    video_file.audio.write_audiofile(f'{file[:-4]}.wav') #Replace '.mp4' extension with '.wav'
    
    return
    
def load_write_transcript(rep,audio_dir,file,file_text,language,start,duration):
    
    r = sr.Recognizer()
    audio_file = sr.AudioFile(f'{rep}/{audio_dir}/{file}')
    with audio_file as source:
        audio = r.record(source,offset=start,duration=duration)
        transcript = r.recognize_google(audio,language=language)    
    
    transcript_file = file_text #Replace '.wav' extension with '.txt
    
    with open(transcript_file,'w') as file:
                    file.write(transcript)

    return


if __name__ == '__main__':

    rep = '/Users/Alex/Cours_Telecom/INFMDI 780/Data'
    video_dir = 'Videos_test'
    audio_dir = 'Audios_test'
    text_dir = 'Texts_test'
    
    language = 'fr-FR'
    
    video_time_df = pd.read_csv('Questions_Time.csv',header=None)
    
    video_files = video_time_df[0].to_list()#os.listdir(f'{rep}/{video_dir}/')
    
    os.chdir(f'{rep}/{audio_dir}/')
    
    for file in video_files:
        
        file_extension = f'{file}.mp4'
        
        #if file[-3:]=='mp4':
        load_write_audio(rep,video_dir,file_extension)
    
            
    audio_files = video_files#os.listdir(f'{rep}/{audio_dir}/')
    
    os.chdir(f'{rep}/{text_dir}/')
    
    for (i,file) in enumerate(audio_files):
        
        file_extension = f'{file}.wav'
        
    
        #if file[-3:]=='wav':
        for j in range(1,9,2):
            
            start = video_time_df.loc[i,j]
            duration = video_time_df.loc[i,j+1] - start
            file_text = f'{file}_{j//2+1}.txt'
            
            load_write_transcript(rep,audio_dir,file_extension,file_text,language,start-0.5,duration+0.5)
