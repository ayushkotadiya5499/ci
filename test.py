from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

df=pd.read_csv('salary_dataset_practice.csv')

X=df.drop('salary',axis=1)
y=df['salary']

num_cols=['experience','age','projects_completed','hours_per_week']
cat_cols=['job_role','education_level','location']

num_pipe=Pipeline(steps=[
    ('scaler',StandardScaler())
])

cat_pipe=Pipeline(steps=[
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

compose=ColumnTransformer(transformers=[
    ('num',num_pipe,num_cols),
    ('cat',cat_pipe,cat_cols)
],remainder='passthrough',verbose_feature_names_out=False)

pipe=Pipeline(steps=[
    ('preprocessor',compose),
    ('model',DecisionTreeRegressor(random_state=42))
])

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe.fit(X_train, y_train)
y_pred=pipe.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))