import numpy as np
import pandas as pd

def convert_start_time(start_time_string):
    """ Convert the start time format in the Google sheet in seconds

    Args:
        start_time_string (string): start time in format MM.SS

    Returns:
        int: start time in seconds
    """
    minutes, seconds = map(int,start_time_string.split('.'))
    return 60 * minutes + seconds

def get_annotations_video(filename_annotations, video_name, agreg_annotators):
    """ Get the annotations from the Google sheet file
        - response 8 start time
        - response 17 start time
        - stress scores
    Args:
        filename_annotations (string): path of the video annotation file
        video_name (string): name of the video (e.g "Video_1")
        agreg_annotators(string): first, mean, max, min, sum | default = max
        agreg_annotations(string): first, mean, max, min | default = max (for column 12-16 & 18-23)

    Returns:
        response8_start_time (int): start time (sec) of response 8 ('' if the video name is not found)
        response17_start_time (int): start time (sec) of response 17 ('' if the video name is not found)
        stress_annotations ([int]): stress scores ('' if the video name is not found)
    """
    df_annotations = pd.read_csv(filename_annotations,  header=None).drop([0, 1, 2, 3])
    df_annotations_video = df_annotations[df_annotations.iloc[:,1] == video_name]

    #Get participant type (formateur vs stagiaire)
    type_candidat = str(df_annotations_video.iloc[0,19])

    #Get sex (H/F)
    sexe = str(df_annotations_video.iloc[0,20])

    # Get Q8 start time
    response8_start_time = str(df_annotations_video.iloc[0,3]).replace(",", ".")
    response8_start_time = convert_start_time(response8_start_time)
    # Get Q17 start time
    response17_start_time = str(df_annotations_video.iloc[0,4]).replace(",", ".")
    response17_start_time = convert_start_time(response17_start_time)

    # Load annotations as array of float
    stress_annotations = np.array(df_annotations_video.iloc[:,5:19].astype(float)) #hardcoded columns location

    # Aggregate annotation of annotators
    if agreg_annotators == 'mean':
        stress_annotations = np.mean(stress_annotations,axis=0)
    elif agreg_annotators == 'sum':
        stress_annotations = np.sum(stress_annotations,axis=0)
    elif agreg_annotators == 'max':
        stress_annotations = np.max(stress_annotations,axis=0)
    elif agreg_annotators == 'min':
        stress_annotations = np.min(stress_annotations,axis=0)

    # Aggregate annotation of columns
    #For diapo 12-16
    res = stress_annotations[5:8].max() #Compute max (can be changed)
    stress_annotations = np.delete(stress_annotations,[5,6,7]) #Delete the cols
    stress_annotations = np.insert(stress_annotations,5,res) #Replace by the aggregate

    #Now we do the same for diapo 18-23
    res = stress_annotations[6:10].max()
    stress_annotations = np.delete(stress_annotations,[6,7,8,9])
    stress_annotations = np.insert(stress_annotations,7,res)

    #Convert stress_annotations from np.array to list
    stress_annotations = stress_annotations.tolist()
 

    return response8_start_time, response17_start_time, stress_annotations, type_candidat, sexe

def add_video_annotations(df_features, filename_annotations, time_column_index, video_name, agreg_annotators):
    """ Add annotations information to the DataFrame of features

    Args:
        df_features (pandas.DataFrame): features
        filename_annotations (string): path of the annotations file
        time_column_index ([int]): index of the time column of the DataFrame
        video_name (string): name of the video to be recorded in the DataFrame

    Returns:
        pandas.DataFrame: DataFrame of features with annotations
                          The DataFrame is empty if annotations were not found
    """

    diapos = [1,8,9,10,11,12,17,18] # hardcoded
    response8_start_time, response17_start_time, stress_annotations, type_candidat, sexe = get_annotations_video(filename_annotations, video_name, agreg_annotators)

    df = df_features.copy()
    df['video_name'] = video_name
    df['stress_global'] = stress_annotations[-1]
    df['type_candidat'] = type_candidat
    df['sexe'] = sexe
    
    #Time in sec of questions 1-8 / 9 / 10 / 11 / 12-16 / 17 / 18-23
    sequences_limits = np.array([-10+response8_start_time, 40+response8_start_time, 139+response8_start_time, 206+response8_start_time, 302+response8_start_time, response17_start_time, response17_start_time+29])
    sequences_limits = list(sequences_limits)

    limit_1 = 0
    i = 0
    for limit_2 in sequences_limits:
        df.loc[(df.iloc[:,time_column_index] >= limit_1) & (df.iloc[:,time_column_index] < limit_2),['stress','diapo']] = [stress_annotations[i],diapos[i]]
        limit_1 = limit_2
        i += 1

    # last sequence
    df.loc[(df.iloc[:,time_column_index] >= limit_2),['stress','diapo']] =  [stress_annotations[i],diapos[i]]
    df.diapo = df.diapo.astype(int)
    df.stress = df.stress.astype(float)
    df.stress_global = df.stress_global.astype(float)

    return df