import pandas as pd
import numpy as np
from sklearn import preprocessing

X = pd.read_csv('main_X.csv')
# X = pd.read_csv('data/train.csv')

########### persentage of survival ########### ########### 
# print(X[['Sex', 'Survived']].groupby('Sex', as_index=False).mean())
# print('-'*20)
# print(X[['Embarked', 'Survived']].groupby('Embarked', as_index=False).mean())
# print('-'*20)
# print(X[['AgeBin', 'Survived']].groupby('AgeBin', as_index=False).mean())

########### ########### ########### ########### ###########


x = X[['Pclass']].values.astype(float)
# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)

# Run the normalizer on the dataframe
df_normalized = pd.DataFrame(x_scaled)
print(df_normalized)
	
