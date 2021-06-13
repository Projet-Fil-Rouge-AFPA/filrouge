#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from annotations import get_annotations_video
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

#video_file = 'https://drive.google.com/drive/folders/1RQCPBOhs7LQokxrioaw0c1fG-mQMLOKt'
#audio_file = 'https://drive.google.com/drive/folders/1SL719K9cKXB4C0mpucy2mc_lwZMSihVn'
#text_file = 'https://drive.google.com/drive/folders/1BREEwBkjulmjpfKQefhx9DiO3jMXBR8s'

folder_Features = '/Users/Alex/Cours_Telecom/INFMDI 780/Data/Features'
video_file = f'{folder_Features}/video_features.p'
audio_file = f'{folder_Features}/audio_emobase_data_X_audio.p'
text_file = f'{folder_Features}/text_features'

filename_annotations ='https://docs.google.com/\
spreadsheets/d/1Rqu1sJiD-ogc4a6R491JTiaYacptOTqh6DKqhwTa8NA/gviz/tq?tqx=out:csv&sheet=Template'

#Merge features
video_feat = pd.read_pickle(video_file)
audio_feat = pd.read_pickle(audio_file)
text_feat = pd.read_pickle(text_file)

multi_feat = pd.concat([video_feat,audio_feat,text_feat],axis=1)

#videos_excluded = ['WIN_20210329_14_13_45_Pro','WIN_20210402_14_27_50_Pro']
#multi_feat = multi_feat.drop(videos_excluded,axis=0)

#Retrieve labels
df_annotations = pd.read_csv(filename_annotations,  header=None).drop([0, 1, 2, 3])

diapos = [1,8,9,10,11,12,17,18]

video_names = set(df_annotations[1].values)

dict = {}

for video_name in video_names:
    
    labels = get_annotations_video(filename_annotations, video_name,'max')[2]
    
    dict_diapo = {}
    
    for (i,diapo) in enumerate(diapos):
        
        dict_diapo[diapo] = labels[i]
    
    dict[video_name] = dict_diapo

df_labels = pd.DataFrame.from_dict(dict,orient='index')
df_labels = df_labels.stack()
#df_labels = df_labels.drop(videos_excluded,axis=0)

#Merge multi_feeatures and labels
data = pd.concat([multi_feat,df_labels],axis=1)
data = data.fillna(0)

#Classifier
clf_RF = RandomForestClassifier(criterion='entropy',random_state=42)

#Leave-One-Interviewer-Out cross-validation
LOGO = LeaveOneGroupOut()

df_group = df_labels.copy()
df_group = df_group.reset_index()
groups = df_group.level_0.values

#Normalising the data
scaler = StandardScaler()
X_cv = scaler.fit_transform(data.drop(0,axis=1))

y_cv = data[0]

score_list_RF = []
score_list_SVC = []

F1_list_RF = []
F1_list_SVC = []

for train_index, test_index in LOGO.split(X_cv,y_cv,groups):
    X_train, X_test = X_cv[train_index], X_cv[test_index]
    y_train, y_test = y_cv[train_index], y_cv[test_index]
    
    #Train fitting
    clf_RF.fit(X_train,y_train)
            
    #Test accuracy and F1
    score_RF = clf_RF.score(X_test,y_test)
    
    score_list_RF.append(score_RF)
    
    y_pred_RF = clf_RF.predict(X_test)
    F1_RF = f1_score(y_test,y_pred_RF,average='weighted')
    
    F1_list_RF.append(F1_RF)

print(f'RF: {round(np.mean(score_list_RF),4)} / {round(np.mean(F1_list_RF),4)}')