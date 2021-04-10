import numpy as np
import pandas as pd

def generate_sequences_boundaries(response8_start_time_in_seconds):
    sequences_limits = np.array([-10, 40, 139, 206, 302, 362, 415, 448, 477, 520, 560, 593])
    sequences_limits = sequences_limits + response8_start_time_in_seconds
    return list(sequences_limits)

def convert_start_time(start_time_string):
    minutes, seconds = map(int,start_time_string.split('.'))
    return 60 * minutes + seconds

def get_annotations_video(filename_annotations, video_name):
    df_annotations = pd.read_csv(filename_annotations,  header=None).drop([0, 1, 2, 3])
    # video_name = 'WIN_20210406_18_35_52_Pro'
    response8_start_time = df_annotations[df_annotations.iloc[:,1] == video_name].iloc[0,3]
    stress = df_annotations[df_annotations.iloc[:,1] == video_name].iloc[0,4:18]
    # we take the first line of the annotations
    # this can be changed
    response8_start_time = df_annotations[df_annotations.iloc[:,1] == video_name].iloc[0,3]
    stress_annotations = df_annotations[df_annotations.iloc[:,1] == video_name].iloc[0,4:18]

    if (len(response8_start_time) == 0) :
        return False # the video_name was not found

    response8_start_time = convert_start_time(response8_start_time)
    stress_annotations = [int(stress) for stress in stress_annotations]
    
    return response8_start_time, stress_annotations

def add_video_annotations(df_features, filename_annotations, time_column_index, video_name):
    
    diapos = [1,8,9,10,11,12,15,16,17,18,19,20,21] # hardcoded
    
    response8_start_time, stress_annotations = get_annotations_video(filename_annotations, video_name)

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
        df.loc[(df.iloc[:,time_column_index] >= limit_1) & (df.iloc[:,time_column_index] < limit_2),['stress','diapo']] =[stress_annotations[i],diapos[i]]
        limit_1 = limit_2
        i += 1
    
    # last sequence
    df.loc[(df.iloc[:,time_column_index] >= limit_2),['stress','diapo']] =[stress_annotations[i],diapos[i]]
    df.diapo = df.diapo.astype(int)
    df.stress = df.stress.astype(int)
    df.stress_global = df.stress_global.astype(int)

    return df
