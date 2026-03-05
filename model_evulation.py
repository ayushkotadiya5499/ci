import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score,root_mean_squared_error,mean_absolute_error
import joblib
import yaml

with open('params.yml') as f:
    params = yaml.safe_load(f)
    test_size = params['data_loading']['test_size']
    n_estimators = params['model_building']['n_estimators']
    min_samples_leaf = params['model_building']['min_samples_leaf']
    max_depth = params['model_building']['max_depth']
    min_samples_split = params['model_building']['min_samples_split']
    random_state=params['data_loading']['random_state']


model=joblib.load('models/model.pkl')

def load_data(file_path):
    df=pd.read_csv(file_path)
    return df

def x_y_split(df,target_column='target'):
    X=df.drop(columns=[target_column])
    y=df[target_column]
    return X,y


df=load_data("/home/web-h-053/intership/mlops/ci-cd/ci/testing/processed_data/processed_train_data.csv")
X,y=x_y_split(df,target_column='target')
predictions=model.predict(X)
mse=mean_squared_error(y,predictions)
r2=r2_score(y,predictions)
rmse=root_mean_squared_error(y,predictions) 
mae=mean_absolute_error(y,predictions)


with open('model_evaluation.txt','a+') as f:
    f.write('===============================\n')
    f.write('Model Evaluation Results:\n')

    f.write(f'Mean Squared Error: {mse}\n')
    f.write(f'R2 Score: {r2}\n')
    f.write(f'Root Mean Squared Error: {rmse}\n')
    f.write(f'Mean Absolute Error: {mae}\n')

    f.write('random state - {} \n'.format(random_state))
    f.write('n_estimators - {} \n'.format(n_estimators))
    f.write('min_samples_leaf - {} \n'.format(min_samples_leaf))
    f.write('max_depth - {} \n'.format(max_depth))
    f.write('min_samples_split - {} \n'.format(min_samples_split))
    f.write('test_size - {} \n'.format(test_size))

    f.write('-----------------------------\n')
    f.write('===============================\n')

print("r2 score:",r2)
print("mean squared error:",mse)
print("root mean squared error:",rmse)
print("mean absolute error:",mae)