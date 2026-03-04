import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
import joblib
import yaml

with open('params.yml') as f:
    params = yaml.safe_load(f)


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df   

def xy_split(df,target_col):
    X=df.drop(target_col,axis=1)
    y=df[target_col]
    return X,y

def build_model(X_train,y_train):
    model=RandomForestRegressor(random_state=params['data_loading']['random_state'],n_estimators=params['model_building']['n_estimators'],max_depth=params['model_building']['max_depth'],min_samples_split=params['model_building']['min_samples_split'])
    model.fit(X_train,y_train)
    return model

df=load_data('preprocessed_data/preprocessed_train_df.csv')
X,y=xy_split(df,'salary')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=params['data_loading']['test_size'],random_state=params['data_loading']['random_state'])
model=build_model(X_train,y_train)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')