import os
import subprocess
import platform
import pandas as pd
import math
from annotations import add_video_annotations
from itertools import repeat

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

def create_dataframe_video(OpenFace_processed_path, Name_video):
    '''
    Create a dataframe from csv extracted with OpenFace
    
    Parameters
    ----------
    OpenFace_processed_path: String
    Path for the directory processed of OpenFace (e.g '/Users/OpenFace/processed/')
    
    Name_video : String
    Name of the video (e.g. 'Video1')
    '''
    df = pd.read_csv(OpenFace_processed_path+Name_video+'.csv') 
    #rename the column erasing the space in the name of each column
    l=[]
    for i in df.columns:
        l.append(i.replace(" ", ""))  
    df.columns =l   
    return df
    

def get_df_video_with_annotations(OpenFace_processed_path, Name_video, Annotations_path, aggreg):
    '''
    Create a dataframe with annotations from csv extracted with OpenFace
    
    Parameters
    ----------
    OpenFace_processed_path: String
    Path for the directory processed of OpenFace (e.g '/Users/OpenFace/processed/')
    
    Name_video : String
    Name of the video without csv (e.g. 'Video1')

    AnnotationPath: String
    Path for the csv with annotations(e.g ''/Users/video_stress/Videos_Annotations - Template.csv'')
    '''
    column_timestamp = 2
    
    df_video = create_dataframe_video(OpenFace_processed_path, Name_video)
    return add_video_annotations(df_video, Annotations_path, column_timestamp, Name_video, aggreg)     
def check_success(df):
   
    '''
    Gives the percentage of the frames in which OpenFace succeeded in the detection a face 
    
    Parameters
    ----------
    df: Dataframe
    Name of the dataframe 
    '''
    return df["success"].value_counts()[1]*100/df.shape[0]

def eliminate_features(df):
    '''
    Takes a dataframe, as produced from the csv of OpenFace, 
    and keeps only the features 'frame', 'face_id', 'timestamp', 'confidence', 'success',
    and the features related to eye gaze, AU, and head movements.

    Parameters
    ----------
    df: Dataframe
    Name of the dataframe
    '''
    for i in df.columns:
        if "eye" in i or "x_" in i or 'y_' in i or 'z_' in i or 'X_' in i or 'Y_' in i or 'Z_' in i or 'p_' in i:
            del df[i]
    return df 

def total_distance_head(df):
    """ Takes a datframe as produced from the csv of OpenFace, 
    and returns the total distance in millimeters traveled by the head
    during the video
    
    Parameters
    ----------
    df: Dataframe
    Name of the dataframe
    """

    dist=0
    for i in range(df.shape[0]-1):
        dist+= math.sqrt((df.loc[i, "pose_Tx"]-df.loc[i+1, "pose_Tx"])**2 +(df.loc[i, "pose_Ty"]-df.loc[i+1, "pose_Ty"])**2+(df.loc[i, "pose_Tz"]-df.loc[i+1, "pose_Tz"])**2 )
    return dist        

def create_df_distances_head(df):
    """Takes a dataframe which has the column diapo and add a column "dist-head". 
    This new column contains the distance traveled by the head during a diapo, therefore
    it is a column which has a numer of values equal to the number of different diapos

    Parameters
    
    """
    df=df.copy()
    L=[]
    diapos = [1,8,9,10,11,12,17, 18]
    for j in diapos:
        lj=df.index[df['diapo'] == j].tolist()
        distj=0
        for i in lj[:-1]:
            distj+= math.sqrt((df.loc[i, "pose_Tx"]-df.loc[i+1, "pose_Tx"])**2 +(df.loc[i, "pose_Ty"]-df.loc[i+1, "pose_Ty"])**2+(df.loc[i, "pose_Tz"]-df.loc[i+1, "pose_Tz"])**2 )
        Lj = list(repeat(distj, len(lj))) 
        L+=Lj
    df["dist_head"]= L   
    return df
def create_df_difference_timestamp(df):    
    """Takes a dataframe which has the column diapo and add a column "duration". 
    This new column contains the duration of each diapo, therefore
    it is a column which has a numer of values equal to the number of different diapos

    Parameters
     ----------
    df: Dataframe
    Name of the dataframe"""
    df=df.copy()
    L=[]
    diapos = [1,8,9,10,11,12,17, 18]
    for j in diapos:
        lj=df.index[df['diapo'] == j].tolist()
        tempj = df.loc[lj[-1],'timestamp']-df.loc[lj[0], 'timestamp']
       
        Lj = list(repeat(tempj, len(lj))) 
        L+=Lj
    df["duration"]= L
    return df

