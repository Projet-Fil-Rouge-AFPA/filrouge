#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import moviepy.editor as mp
import speech_recognition as sr

def load_write_audio(rep,video_dir,file):
    
    video_file = mp.VideoFileClip(f'{rep}/{video_dir}/{file}')
    video_file.audio.write_audiofile(f'{file[:-4]}.wav') #Replace '.mp4' extension with '.wav'
    
    return
    
def load_write_transcript(rep,audio_dir,file,language,start,duration):
    
    r = sr.Recognizer()
    audio_file = sr.AudioFile(f'{rep}/{audio_dir}/{file}')
    with audio_file as source:
        audio = r.record(source,offset=start,duration=duration)
        transcript = r.recognize_google(audio,language=language)    
    
    transcript_file = file[:-4]+'.txt' #Replace '.wav' extension with '.txt
    
    with open(transcript_file,'w') as file:
                    file.write(transcript)

    return


if __name__ == '__main__':

    rep = '/Users/Alex/Cours_Telecom/INFMDI 780/Data'
    video_dir = 'Videos_test'
    audio_dir = 'Audios_test'
    text_dir = 'Texts_test'
    
    language = 'fr-FR'
    start = 160
    duration = 60
    
    video_files = os.listdir(f'{rep}/{video_dir}/')
    
    os.chdir(f'{rep}/{audio_dir}/')
    
    for file in video_files:
        
        if file[-3:]=='mp4':
            load_write_audio(rep,video_dir,file)
            
    
    audio_files = os.listdir(f'{rep}/{audio_dir}/')
    
    os.chdir(f'{rep}/{text_dir}/')
    
    for file in audio_files:
        
        if file[-3:]=='wav':
            load_write_transcript(rep,audio_dir,file,language,start-1,duration+1)
    

    