import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


def test_salary_pipeline_r2():
    # load the salary dataset from repository
    df = pd.read_csv('salary_dataset_practice.csv')
    X = df.drop('salary', axis=1)
    y = df['salary']

    num_cols = ['experience', 'age', 'projects_completed', 'hours_per_week']
    cat_cols = ['job_role', 'education_level', 'location']

    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

    compose = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    pipe = Pipeline(steps=[
        ('preprocessor', compose),
        ('model', DecisionTreeRegressor(random_state=12,max_depth=15,min_samples_leaf=5,min_samples_split=10))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    score = r2_score(y_test, y_pred)
    # expect the model to capture some variance in salary data
    assert score >= 0.5, f"R2 too low: {score}"
