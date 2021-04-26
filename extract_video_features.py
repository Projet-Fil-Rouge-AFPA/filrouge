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

def create_df_difference_timestamp(df):    
    """Takes a dataframe which has the column diapo and adds a column "duration". 
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

def create_df_distances_head(df):
    """Takes a dataframe which has the column "diapo" and the column "duration" and adds a column "dist-head". 
    This new column contains the distance traveled by the head during a diapo, divided by the dutration of each diapo, therefore
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
    df["dist_head_total"]= L   
    #df["dist_head"]= df["dist_head_total"]/df["duration"]
    df["dist_head"]= df["dist_head_total"]/1
    df.drop("dist_head_total",axis='columns', inplace=True)
    return df

def create_df_distances_gaze(df):
    """Takes a dataframe which has the column "diapo" and the column "duration" and adds a column "dist-gaze_0" and "dist-gaze_1". 
    This two new columns contain the distance traveled by each of the vector associated to the gaze, divided by the duration of the diapo.

    Parameters
    ----------
    df: Dataframe
    Name of the dataframe
    
    """
    df=df.copy()
    L0=[]
    diapos = [1,8,9,10,11,12,17, 18]
    for j in diapos:
        lj=df.index[df['diapo'] == j].tolist()
        distj=0
        for i in lj[:-1]:
            distj+= math.sqrt((df.loc[i, "gaze_0_x"]-df.loc[i+1, "gaze_0_x"])**2 +(df.loc[i, "gaze_0_y"]-df.loc[i+1, "gaze_0_y"])**2+(df.loc[i, "gaze_0_z"]-df.loc[i+1, "gaze_0_z"])**2 )
        Lj = list(repeat(distj, len(lj))) 
        L0+=Lj
    df["dist_gaze_0_total"]= L0 
    #df["dist_gaze_0"]= df["dist_gaze_0_total"]/df["duration"]
    df["dist_gaze_0"]= df["dist_gaze_0_total"]/1

    L1=[]
    diapos = [1,8,9,10,11,12,17, 18]
    for j in diapos:
        lj=df.index[df['diapo'] == j].tolist()
        distj=0
        for i in lj[:-1]:
            distj+= math.sqrt((df.loc[i, "gaze_1_x"]-df.loc[i+1, "gaze_1_x"])**2 +(df.loc[i, "gaze_1_y"]-df.loc[i+1, "gaze_1_y"])**2+(df.loc[i, "gaze_1_z"]-df.loc[i+1, "gaze_1_z"])**2 )
        Lj = list(repeat(distj, len(lj))) 
        L1+=Lj
    df["dist_gaze_1_total"]= L1 
    #df["dist_gaze_1"]= df["dist_gaze_1_total"]/df["duration"]
    df["dist_gaze_1"]= df["dist_gaze_1_total"]/1

    df.drop(['dist_gaze_1_total', 'dist_gaze_0_total'],axis='columns', inplace=True)  
    return df


def create_df_distances_pose_x(df):
    """Takes a dataframe which has the column "diapo" and the column "duration" and adds a column "dist_pose_x". 
    This new column contains the variation of pose_x during each diapo, divided by the duration of the diapo.

    Parameters
    ----------
    df: Dataframe
    Name of the dataframe
    """
    L=[]
    diapos = [1,8,9,10,11,12,17, 18]
    for j in diapos:
        lj=df.index[df['diapo'] == j].tolist()
        distj=0
        for i in lj[:-1]:
            distj+= abs(df.loc[i, "pose_Rx"]-df.loc[i+1, "pose_Rx"])
        Lj = list(repeat(distj, len(lj))) 
        L+=Lj
    df["pose_x_total"]= L
    #df["dist_pose_x"]=df["pose_x_total"]/df["duration"]
    df["dist_pose_x"]=df["pose_x_total"]/1
    df.drop('pose_x_total',axis='columns', inplace=True)
    return df

def create_df_distances_pose_y(df):
    """Takes a dataframe which has the column "diapo" and the column "duration" and adds a column "dist_pose_y". 
    This new column contains the variation of pose_y during each diapo, divided by the duration of the diapo.

    Parameters
    ----------
    df: Dataframe
    Name of the dataframe
    
    """
    L=[]
    diapos = [1,8,9,10,11,12,17, 18]
    for j in diapos:
        lj=df.index[df['diapo'] == j].tolist()
        distj=0
        for i in lj[:-1]:
            distj+= abs(df.loc[i, "pose_Ry"]-df.loc[i+1, "pose_Ry"])
        Lj = list(repeat(distj, len(lj))) 
        L+=Lj
    df["pose_y_total"]= L 
    #df["dist_pose_y"]=df["pose_y_total"]/df["duration"] 
    df["dist_pose_y"]=df["pose_y_total"]/1
    df.drop('pose_y_total',axis='columns', inplace=True) 
    return df   

def create_df_distances_pose_z(df):
    """Takes a dataframe which has the column "diapo" and the column "duration" and adds a column "pose_z". 
    This new column contains the variation of pose_z during each diapo, divided by the duration of the diapo.


    Parameters
    ----------
    df: Dataframe
    Name of the dataframe
    
    """
    L=[]
    diapos = [1,8,9,10,11,12,17, 18]
    for j in diapos:
        lj=df.index[df['diapo'] == j].tolist()
        distj=0
        for i in lj[:-1]:
            distj+= abs(df.loc[i, "pose_Rz"]-df.loc[i+1, "pose_Rz"])
        Lj = list(repeat(distj, len(lj))) 
        L+=Lj
    df["pose_z_total"]= L  
    #df["dist_pose_z"]=df["pose_z_total"]/df["duration"]
    df["dist_pose_z"]=df["pose_z_total"]/1

    df.drop('pose_z_total',axis='columns', inplace=True)  
    return df


def add_dist_features(df):
    """Takes a dataframe, adds all the features with distances and erase the position features without distances. 
    This new column contains the variation of pose_z during each diapo, divided by the duration of the diapo.


    Parameters
    ----------
    df: Dataframe
    Name of the dataframe
    
    """


    df =create_df_difference_timestamp(df)
    df = create_df_distances_head(df)
    df = create_df_distances_gaze(df)
    df= create_df_distances_pose_x(df)
    df= create_df_distances_pose_y(df)
    df=create_df_distances_pose_z(df)

    features_to_erase = ['gaze_0_x',
       'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
       'gaze_angle_x', 'gaze_angle_y', 'pose_Tx', 'pose_Ty', 'pose_Tz',
       'pose_Rx', 'pose_Ry', 'pose_Rz']
    df.drop(features_to_erase,axis='columns', inplace=True)
    return df    


