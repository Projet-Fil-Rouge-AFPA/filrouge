import os
import subprocess
import platform
import pandas as pd
import math
from annotations import add_video_annotations


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
    Give the percentage of the frames in which OpenFace succeeded in the detection a face 
    
    Parameters
    ----------
    df: Dataframe
    Name of the dataframe 
    '''
    return df["success"].value_counts()[1]*100/df.shape[0]

def eliminate_features(df):
    '''
    Take a dataframe, as produced from the csv of OpenFace, 
    and keep only the features 'frame', 'face_id', 'timestamp', 'confidence', 'success',
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

def create_df_difference_timestamp(df):    
    """
    Take a dataframe which has the column 'diapo' and add a column 'duration'. 
    This new column contains the duration of each diapo, therefore
    it is a column which has a number of values equal to the number of different diapos

    Parameters
    ----------
    df: Dataframe
    Name of the dataframe
    """
    df=df.copy()
    diapos = [1,8,9,10,11,12,17, 18]
    for j in diapos:
        lj=df.index[df['diapo'] == j].tolist()
        tempj = df.loc[lj[-1],'timestamp']-df.loc[lj[0], 'timestamp']
        df.loc[df['diapo']==j,'duration']=tempj
       
    return df                 

def create_df_distances_head(df):
    """
    Take a dataframe which has the column "diapo" and the column and add a column "dist-head". 
    This new column contains the distance traveled by the head during a diapo, therefore
    it is a column which has a numer of values equal to the number of different diapos

    Parameters
    ----------
    df: Dataframe
    Name of the dataframe
    """

    df=df.copy()
    diapos = [1,8,9,10,11,12,17, 18]
    for j in diapos:
        lj=df.index[df['diapo'] == j].tolist()
        distj=0
        for i in lj[:-1]:
            distj+= math.sqrt((df.loc[i, "pose_Tx"]-df.loc[i+1, "pose_Tx"])**2 +(df.loc[i, "pose_Ty"]-df.loc[i+1, "pose_Ty"])**2+(df.loc[i, "pose_Tz"]-df.loc[i+1, "pose_Tz"])**2 )

        df.loc[df['diapo']==j,'dist_head']=distj    
    df['dist_head']=df['dist_head']/df['duration']     
    return df

def create_df_distances_gaze(df):
    """
    Take a dataframe which has the column "diapo" and the column "duration" and add a column "dist-gaze_0" and "dist-gaze_1". 
    This two new columns contain the distance traveled by each of the vector associated to the gaze.

    Parameters
    ----------
    df: Dataframe
    Name of the dataframe
    
    """
    df=df.copy()
    diapos = [1,8,9,10,11,12,17, 18]
    for j in diapos:
        lj=df.index[df['diapo'] == j].tolist()
        distj=0
        for i in lj[:-1]:
            distj+= math.sqrt((df.loc[i, "gaze_0_x"]-df.loc[i+1, "gaze_0_x"])**2 +(df.loc[i, "gaze_0_y"]-df.loc[i+1, "gaze_0_y"])**2+(df.loc[i, "gaze_0_z"]-df.loc[i+1, "gaze_0_z"])**2 )
        df.loc[df['diapo']==j,'dist_gaze_0']=distj 
    df['dist_gaze_0']=df['dist_gaze_0']/df['duration'] 
    diapos = [1,8,9,10,11,12,17, 18]
    for j in diapos:
        lj=df.index[df['diapo'] == j].tolist()
        distj=0
        for i in lj[:-1]:
            distj+= math.sqrt((df.loc[i, "gaze_1_x"]-df.loc[i+1, "gaze_1_x"])**2 +(df.loc[i, "gaze_1_y"]-df.loc[i+1, "gaze_1_y"])**2+(df.loc[i, "gaze_1_z"]-df.loc[i+1, "gaze_1_z"])**2 )
        df.loc[df['diapo']==j,'dist_gaze_1']=distj
    df['dist_gaze_1']=df['dist_gaze_1']/df['duration']
    return df


def create_df_distances_pose_x(df):
    """
    Take a dataframe which has the column "diapo" and the column "duration" and add a column "dist_pose_x". 
    This new column contains the variation of pose_x during each diapo.

    Parameters
    ----------
    df: Dataframe
    Name of the dataframe
    """
    diapos = [1,8,9,10,11,12,17, 18]
    for j in diapos:
        lj=df.index[df['diapo'] == j].tolist()
        distj=0
        for i in lj[:-1]:
            distj+= abs(df.loc[i, "pose_Rx"]-df.loc[i+1, "pose_Rx"])
        df.loc[df['diapo']==j,'dist_pose_x']=distj  
    df['dist_pose_x']=  df['dist_pose_x']/df['duration']
    return df

def create_df_distances_pose_y(df):
    """
    Take a dataframe which has the column "diapo" and the column "duration" and add a column "dist_pose_y". 
    This new column contains the variation of pose_y during each diapo.

    Parameters
    ----------
    df: Dataframe
    Name of the dataframe
    
    """
   
    diapos = [1,8,9,10,11,12,17, 18]
    for j in diapos:
        lj=df.index[df['diapo'] == j].tolist()
        distj=0
        for i in lj[:-1]:
            distj+= abs(df.loc[i, "pose_Ry"]-df.loc[i+1, "pose_Ry"])
        df.loc[df['diapo']==j,'dist_pose_y']=distj
    df['dist_pose_y']=  df['dist_pose_y']/df['duration']    
    return df   

def create_df_distances_pose_z(df):
    """
    Take a dataframe which has the column "diapo" and the column "duration" and add a column "pose_z". 
    This new column contains the variation of pose_z during each diapo.


    Parameters
    ----------
    df: Dataframe
    Name of the dataframe
    
    """
   
    diapos = [1,8,9,10,11,12,17, 18]
    for j in diapos:
        lj=df.index[df['diapo'] == j].tolist()
        distj=0
        for i in lj[:-1]:
            distj+= abs(df.loc[i, "pose_Rz"]-df.loc[i+1, "pose_Rz"])
        df.loc[df['diapo']==j,'dist_pose_z']=distj
    df['dist_pose_z']=  df['dist_pose_z']/df['duration']     
    return df


def add_dist_features(df):
    """
    Take a dataframe, add all the features with distances and erase the position features without distances.

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

    features_to_erase = ['duration, ''gaze_0_x',
       'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
       'gaze_angle_x', 'gaze_angle_y', 'pose_Tx', 'pose_Ty', 'pose_Tz',
       'pose_Rx', 'pose_Ry', 'pose_Rz']
    df.drop(features_to_erase,axis='columns', inplace=True)
    return df    


