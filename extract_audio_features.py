import numpy as numpy
import pandas as pd 
import matplotlib.pyplot as plt 
import moviepy.editor as mp
import pathlib
import subprocess
import time
from annotations import add_video_annotations

def concatenate_spread(df, col, suffix='_de', lag = 1):
    df[col + suffix] = df[col] - df[col].shift(lag)
    return df

def concatenate_spread_1_and_2(df, col, suffix='_de', lag = 1):
    df[col + suffix] = df[col] - df[col].shift(lag)
    df[col + suffix + suffix] = df[col + suffix] - df[col + suffix].shift(lag)
    return df


def get_videos(directory_path):
    """ Get videos paths

    Args:
        directory_path (string): directory where to search for the videos

    Returns:
        [string]: list of the video paths
        [string]: list of the video names
    """
    # define the path
    currentDirectory = pathlib.Path(directory_path)

    # videos extension
    currentPattern = "*.mp4"

    videos_paths_list = [str(currentFile) for currentFile in currentDirectory.glob(currentPattern)]
    if videos_paths_list:
        video_filenames_list = [video_path.replace(directory_path,'').replace('/','')[:-4] for video_path in videos_paths_list]
    else:
        video_filenames_list = []
        
    return videos_paths_list, video_filenames_list

def extract_audio(video_path):
    """ Extract the audio band from a video

    Args:
        video_path (string): video path

    Returns:
        string: path of the wav audio band
    """ 

    video_file = mp.VideoFileClip(video_path)
    wav_path_filename = video_path + '.wav'
    video_file.audio.write_audiofile(wav_path_filename)
    
    return wav_path_filename

def extract_features(wav_filepath, open_smile_conf_path, feature_name):
    """ Extract features from a wav audio file

    Args:
        wav_filepath (string): path of the wav file
        open_smile_conf_path (string): path of the .conf file of OpenSmile
        feature_name (string): audio features group name

    Returns:
        string: path of the feature csv file
    """

    timeout = 300_000
    try:
        wav_filepath = wav_filepath.replace(' ', '\ ')
        command = "./SMILExtract -C " + open_smile_conf_path + " -I " + wav_filepath + " -O " + wav_filepath + "." + feature_name + ".csv"
        #command = "pwd"
        process = subprocess.run([command], timeout=timeout, shell=True, check=True, capture_output=True)

    except subprocess.TimeoutExpired:
        print("error, timeout =",timeout,"s expired")
        return False
    except Exception as autre_exception:
        print(autre_exception)

    return wav_filepath + "." + feature_name + ".csv"


def extract_features_prosody(wav_filepath):
    """ Extract the prosody features from a wav audio file

    Args:
        wav_filepath (string): path of the wav file

    Returns:
        string: path of the feature csv file
    """
    
    open_smile_conf_path = "filrouge/OpenSmile_config/prosodyAcf_modif.conf"
    feature_name = "prosody"
    extract_features(wav_filepath, open_smile_conf_path, feature_name)

    return wav_filepath + "." + feature_name + ".csv"

def extract_features_mfcc(wav_filepath):
    """ Extract the MFCC features from a wav audio file

    Args:
        wav_filepath (string): path of the wav file

    Returns:
        string: path of the feature csv file
    """

    open_smile_conf_path = "filrouge/OpenSmile_config/MFCC12_0_D_A_Z_modif.conf"
    feature_name = "mfcc"
    extract_features(wav_filepath, open_smile_conf_path, feature_name)

    return wav_filepath + "." + feature_name + ".csv"

def extract_features_emobase(wav_filepath):
    """ Extract the EMOBASE features from a wav audio file

    Args:
        wav_filepath (string): path of the wav file

    Returns:
        string: path of the feature csv file
    """
    
    open_smile_conf_path = "filrouge/OpenSmile_config/emobase_modif.conf"
    feature_name = "emobase"
    extract_features(wav_filepath, open_smile_conf_path, feature_name)
    
    return wav_filepath + "." + feature_name + ".csv"

def extract_features_eGeMAPS(wav_filepath):
    """ Extract the EMOBASE features from a wav audio file

    Args:
        wav_filepath (string): path of the wav file

    Returns:
        string: path of the feature csv file
    """
    
    # open_smile_conf_path = "filrouge/OpenSmile_config/eGeMAPSv01b_modif.conf"
    open_smile_conf_path = "Documents/lib/opensmile/config/egemaps/v01b/eGeMAPSv01b_modif.conf"
    feature_name = "eGeMAPS"
    extract_features(wav_filepath, open_smile_conf_path, feature_name)
    
    filename_output = wav_filepath + "." + feature_name + ".csv"
    df = pd.read_csv(filename_output, delimiter=';')
    df = df.drop(df.columns[0],axis=1)
    for col in df.columns[2:]:
        df = concatenate_spread_1_and_2(df, col)
    df.to_csv(filename_output, sep=';')
    print(filename_output)
    return filename_output

def write_audio_annotations(features_filename_path_list, filename_annotations, agreg_annotators):
    """ Add video annotations to the features file

    Args:
        features_filename_path_list ([string]): list of the features csv files
        filename_annotations (string): path of the csv file of the video annotations (originally a Google Sheet)

    Returns:
        [string]: list of the path of the features files with annotations
    """

    annotated_files_list = []
    video_filenames_list = [features_filename_path.rsplit('/',1)[1].split('.',1)[0] for features_filename_path in features_filename_path_list]

    for features_filename_path, video_filename in zip(features_filename_path_list,video_filenames_list):
        df_features = pd.read_csv(features_filename_path, delimiter=';')
        df_features_annoted = add_video_annotations(df_features, filename_annotations, 1, video_filename, agreg_annotators)
        print(video_filename, df_features_annoted.shape)
        if df_features_annoted.shape[0] > 0:
            features_filename_path = features_filename_path[:-4] + '.annotated.csv'
            df_features_annoted.to_csv(features_filename_path, sep=';', index=False, header=True)
            annotated_files_list.append(features_filename_path)
    
    return annotated_files_list


def preprocess_videos(directory_path, filename_annotations, agreg_annotators):
    """ main function for prepressing videos
        and extracting audio features

    Args:
        directory_path (string): directory path of the videos
        filename_annotations ([type]): path of the csv file of the video annotations (originally a Google Sheet)

    Returns:
        bool: True
    """

    t0 = time.time()
    
    # get videos files
    videos_paths_list, _ = get_videos(directory_path)
    

    # extract the audio bands
    wav_filename_path_list = [extract_audio(video_path) for video_path in videos_paths_list]
    t1 = time.time()
    log = 'Audio bands extraction : ' + str((t1-t0)*1000) + ' ms'
    print(log)

    # extract audio features
    features_filename_path_list = [extract_features_emobase(wav_filename_path) for wav_filename_path in wav_filename_path_list]
    t2 = time.time()
    log = 'Audio features extraction : ' + str((t2-t1)*1000) + ' ms'
    print(log)

    # add video annotations
    annotated_files_list = write_audio_annotations(features_filename_path_list, filename_annotations, agreg_annotators)
    print(annotated_files_list)

    t3 = time.time()
    log = 'Add annotations : ' + str((t3-t2)*1000) + ' ms'
    print(log)
    log = 'Total time : ' + str((t3-t0)*1000) + ' ms'
    print(log)
    
    return True


if __name__ == '__main__':
    
    # arguments initialisation
    # directory_path = '../04_-_Dev/videos'
    directory_path = '/home/neo/Documents/Telecom Paris/INF780 - Projet fil rouge/04_-_Dev/videos'
    filename_annotations = 'Videos_Annotations - Template.csv'
    filename_annotations = 'https://docs.google.com/spreadsheets/d/1Rqu1sJiD-ogc4a6R491JTiaYacptOTqh6DKqhwTa8NA/gviz/tq?tqx=out:csv&sheet=Template'
    agreg_annotators = 'max'
    
    # launching the audio features extractions
     #preprocess_videos(directory_path, filename_annotations, agreg_annotators)

    #extract features
    # currentDirectory = pathlib.Path(directory_path)
    # currentPattern = "*.wav"
    # wav_filename_path_list = [str(currentFile) for currentFile in currentDirectory.glob(currentPattern)]
    # print(wav_filename_path_list)
    # [extract_features_eGeMAPS(wav_filename_path) for wav_filename_path in wav_filename_path_list]

    # write annotations
    currentDirectory = pathlib.Path(directory_path)
    currentPattern = "*.emobase.csv" #"*.emobase.csv"
    features_filename_path_list = [str(currentFile) for currentFile in currentDirectory.glob(currentPattern)]
    print(features_filename_path_list)
    write_audio_annotations(features_filename_path_list, filename_annotations, agreg_annotators)