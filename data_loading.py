import pandas as pd
from sklearn.model_selection import train_test_split
import os 
import yaml


with open('params.yml') as f:
    params = yaml.safe_load(f)


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def split(df,test_size=params['data_loading']['test_size'],random_state=params['data_loading']['random_state']):
    train_df, test_df = train_test_split(df,test_size=test_size,random_state=random_state)
    return train_df, test_df


os.makedirs('data', exist_ok=True)

df=load_data('salary_dataset_practice.csv')
train_df, test_df = split(df,test_size=params['data_loading']['test_size'],random_state=params['data_loading']['random_state'])

train_df.to_csv('data/train_df.csv',index=False)
test_df.to_csv('data/test_df.csv',index=False)