import os
import subprocess
import platform
import pandas as pd

def extract_video_features(OpenFace_directory, Video_path):
    '''
    Extract video features (csv) will be located in the 'processed' folder of OpenFace
    
    Parameters
    ----------
    OpenFace_directory : String
    Directory where the file FeatureExtraction can be found
    
    Video_path : String
    Directory where the Video(e.g. mp4) file can be found
    '''
    
    OS = platform.system() #Returns OS name
    os.chdir(f'{OpenFace_directory}') #Move to the OpenFace directory
    
    if OS == 'Windows':
        cmd = f"FeatureExtraction.exe -f {Video_path} -2Dfp -3Dfp -pdmparams -pose -aus -gaze"
        
    elif OS == 'Darwin':
        cmd = f"build/bin/FeatureExtraction -f {Video_path} -2Dfp -3Dfp -pdmparams -pose -aus -gaze"
        
    elif OS == 'Linux':
        cmd = f"FeatureExtraction -f {Video_path} -2Dfp -3Dfp -pdmparams -pose -aus -gaze"
        
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)

def create_dataframe_video(OpenFace_processed_path, Name_csv):
    '''
    Create a dataframe from csv extracted with OpenFace
    
    Parameters
    ----------
    OpenFace_processed_path: String
    Path for the directory processed of OpenFace (e.g '/Users/OpenFace/processed/')
    
    Name_csv : String
    Name of the csv (e.g. 'Video1.csv')
    '''
    return pd.read_csv(OpenFace_processed_path+Name_csv)