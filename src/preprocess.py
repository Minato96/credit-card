import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(path):
    '''Loads the data and returns Dataframe'''
    return pd.read_csv(path)

def scale_features(df):
    '''Scales 'Time' and 'Amount' columns '''
    scaler =  StandardScaler()

    df[['Time','Amount']] = scaler.fit_transform(df[['Time','Amount']])
    return df,scaler

def get_features_and_target(df,target_col):
    '''Seperates the input feautres and target column'''
    X = df.drop(target_col,axis=1)
    y = df[target_col]
    return X,y

def apply_smote(X,y):
    '''Removes the imabalance in class'''
    print("Original class distribution:", y.value_counts())
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("Resampled class distribution:", y_resampled.value_counts())
    return X_resampled, y_resampled