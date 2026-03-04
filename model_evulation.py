import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,root_mean_squared_error
import joblib
import yaml

with open('params.yml') as f:
    params = yaml.safe_load(f)

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def x_y_split(df,target_col):
    X=df.drop(target_col,axis=1)
    y=df[target_col]
    return X,y

def load_model(model_path):
    model = joblib.load(model_path)
    return model

df=load_data('preprocessed_data/preprocessed_train_df.csv')
X,y=x_y_split(df,'salary')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=params['data_loading']['test_size'],random_state=params['data_loading']['random_state'])
model=load_model('models/model.pkl')

model.fit(X_train,y_train)
y_pred=model.predict(X_test)


with open('model_evaluation.txt', 'w') as f:
    f.write(f"R2 Score: {r2_score(y_test, y_pred)}\n")
    f.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}\n")
    f.write(f"Root Mean Squared Error: {root_mean_squared_error(y_test, y_pred)}\n")
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", root_mean_squared_error(y_test, y_pred))