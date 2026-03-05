import pandas as pd
from sklearn.model_selection import train_test_split
import os 
import yaml

with open('params.yml') as f:
    params = yaml.safe_load(f)
    
def load_data(file_path):
    df=pd.read_csv(file_path)
    return df

def split_data(df,test_size=params['data_loading']['test_size'],random_state=params['data_loading']['random_state']):
    train_df,test_df = train_test_split(df,test_size=test_size,random_state=random_state)
    return train_df,test_df



df=load_data('/home/web-h-053/intership/mlops/ci-cd/ci/testing/data.csv')
train_df,test_df=split_data(df,test_size=params['data_loading']['test_size'],random_state=params['data_loading']['random_state'])
os.makedirs('data',exist_ok=True)

train_df.to_csv('data/train_data.csv',index=False)
test_df.to_csv('data/test_data.csv',index=False)
