from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import pandas as pd 
x,y = make_regression(n_samples=100, n_features=20, noise=0.1)


df=pd.DataFrame(x,columns=[f'feature_{i}' for i in range(x.shape[1])])
df['target']=y

df.to_csv('data.csv',index=False)