import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os 
import joblib
import yaml

with open('params.yml') as f:
    params = yaml.safe_load(f)

def load_data(file_path):
    df=pd.read_csv(file_path)
    return df

def x_y_split(df,target_column='target'):
    X=df.drop(columns=[target_column])
    y=df[target_column]
    return X,y


def train_model(X,y):
    model=RandomForestRegressor(n_estimators=params['model_building']['n_estimators'],random_state=params['data_loading']['random_state'],min_samples_leaf=params['model_building']['min_samples_leaf'],max_depth=params['model_building']['max_depth'],min_samples_split=params['model_building']['min_samples_split'])
    model.fit(X,y)
    return model

df=load_data("/home/web-h-053/intership/mlops/ci-cd/ci/testing/processed_data/processed_train_data.csv")
X,y=x_y_split(df,target_column='target')
model=train_model(X,y)

os.makedirs('models',exist_ok=True)
joblib.dump(model,'models/model.pkl')