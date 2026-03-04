from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os 

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def x_y_split(df:pd.DataFrame, target_col='salary'):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

def preprocess_data(X:pd.DataFrame):
    num_cols = ['experience', 'age', 'projects_completed', 'hours_per_week']
    cat_cols = ['job_role', 'education_level', 'location']

    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore',sparse_output=False))])

    compose = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    return compose.fit_transform(X),compose

def final_data(x,y,compose):
    df=pd.concat([pd.DataFrame(x,columns=compose.get_feature_names_out()),pd.DataFrame(y)], axis=1)
    return df

df=load_data('data/train_df.csv')
X, y = x_y_split(df,target_col='salary')
X_preprocessed,compose = preprocess_data(X)

os.makedirs('preprocesser', exist_ok=True)
joblib.dump(compose, 'preprocesser/X_preprocessed.pkl')
final_df = final_data(X_preprocessed, y, compose)

os.makedirs('preprocessed_data', exist_ok=True)
final_df.to_csv('preprocessed_data/preprocessed_train_df.csv', index=False)