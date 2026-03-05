import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import os
import joblib
import yaml


def load_data(file_path):
    df=pd.read_csv(file_path)
    return df

def x_y_split(df,target_column='target'):
    X=df.drop(columns=[target_column])
    y=df[target_column]
    return X,y


def preprocess_data(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ])
    
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed,preprocessor

def final_df(X_preprocessed,y):
    final_df=pd.DataFrame(X_preprocessed,columns=[f'feature_{i}' for i in range(X_preprocessed.shape[1])])
    final_df['target']=y.values
    return final_df

df=load_data("/home/web-h-053/intership/mlops/ci-cd/ci/testing/data/train_data.csv")
X,y=x_y_split(df,target_column='target')
X_preprocessed,preprocessor=preprocess_data(X)

os.makedirs('preprocessor',exist_ok=True)
joblib.dump(preprocessor,'preprocessor/preprocessor.pkl')

final_df=final_df(X_preprocessed,y)

os.makedirs('processed_data',exist_ok=True)
final_df.to_csv('processed_data/processed_train_data.csv',index=False)