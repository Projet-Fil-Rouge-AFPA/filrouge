import numpy as np
import pandas as pd

def generate_sequences_boundaries(response8_start_time_in_seconds):
    """ Generates the video sequence boundaries

    Args:
        response8_start_time_in_seconds (int): start time of response 8

    Returns:
        [int]: list of sequence start times
    """

    sequences_limits = np.array([-10, 40, 139, 206, 302, 362, 415, 448, 477, 520, 560, 593])
    sequences_limits = sequences_limits + response8_start_time_in_seconds
    return list(sequences_limits)

def convert_start_time(start_time_string):
    """ Convert the start time format in the Google sheet in seconds

    Args:
        start_time_string (string): start time in format MM.SS

    Returns:
        int: start time in seconds
    """

    minutes, seconds = map(int,start_time_string.split('.'))
    return 60 * minutes + seconds

def get_annotations_video(filename_annotations, video_name, operator='first'):
    """ Get the annotations from the Google sheet file
        - response 8 start time
        - stress scores

    Args:
        filename_annotations (string): path of the video annotation file
        video_name (string): name of the video (e.g "Video_1")
        operator(string): first, mean, max, min, sum | default = first

    Returns:
        response8_start_time (int): start time of response 8 ('' if the video name is not found)
        stress_annotations ([int]): stress scores ('' if the video name is not found)
    """
    df_annotations = pd.read_csv(filename_annotations,  header=None).drop([0, 1, 2, 3])
    df_annotations_video = df_annotations[df_annotations.iloc[:,1] == video_name]

    try:

        # Get Q8 start time
        response8_start_time = df_annotations_video.iloc[0,3]
        response8_start_time = convert_start_time(response8_start_time)

        # Get Q17 start time
        response17_start_time = df_annotations_video.iloc[0,4]
        response17_start_time = convert_start_time(response17_start_time)

        # Load annotations as array of float
        stress_annotations_1 = np.array(df_annotations_video.iloc[0,4:18].astype(float)) #hardcoded
        stress_annotations_2 = np.array(df_annotations_video.iloc[1,4:18].astype(float)) #hardcoded

        # Aggregate annotation
        if operator == 'first':
            stress_annotations = stress_annotations_1
        elif operator == 'mean':
            stress_annotations = (stress_annotations_1 + stress_annotations_2) / 2
        elif operator == 'sum':
            stress_annotations = stress_annotations_1 + stress_annotations_2
        elif operator == 'max':
            stress_annotations = np.maximum(stress_annotations_1,stress_annotations_2)
        elif operator == 'min':
            stress_annotations = np.minimum(stress_annotations_1,stress_annotations_2)

    #Error handling
    except IndexError as ex:
        print("Video name is not found in the annotation file", video_name, filename_annotations)
        return '', '' # the video_name was not found

    if (pd.isna(response8_start_time)) or (stress_annotations_1.isna().any()) :
        print("The video was not found in the annotation file or is not fully annotated ", 
                filename_annotations, video_name)
        return '', '' # the video_name was not found

    #Convert stress_annotations from np.array to list
    stress_annotations = stress_annotations.tolist()

    return response8_start_time, stress_annotations

def add_video_annotations(df_features, filename_annotations, time_column_index, video_name):
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

    #A CHANGER
    diapos = [1,8,9,10,11,12,15,16,17,18,19,20,21] # hardcoded
    
    response8_start_time, stress_annotations = get_annotations_video(filename_annotations, video_name)
    if response8_start_time == '': 
        # the video was not found
        return pd.DataFrame()

    df = df_features.copy()
    df['video_name'] = video_name
    df['stress_global'] = stress_annotations[-1]
    # df_list = []
    
    limit_1 = 0
    i = 0
    
    sequences_boundaries = generate_sequences_boundaries(response8_start_time)
    for limit_2 in sequences_boundaries:
        # print(limit_1, limit_2)
        # df_list.append(df[(df['frameTime'] >= limit_1) & (df['frameTime'] < limit_2)])
        df.loc[(df.iloc[:,time_column_index] >= limit_1) & (df.iloc[:,time_column_index] < limit_2),['stress','diapo']] = [stress_annotations[i],diapos[i]]
        limit_1 = limit_2
        i += 1
    
    # last sequence
    df.loc[(df.iloc[:,time_column_index] >= limit_2),['stress','diapo']] =  [stress_annotations[i],diapos[i]]
    df.diapo = df.diapo.astype(int)
    df.stress = df.stress.astype(int)
    df.stress_global = df.stress_global.astype(int)

    return df