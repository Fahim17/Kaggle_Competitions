import pandas as pd
import numpy as np
#from sklearn import preprocessing
#import matplotlib.pyplot as plt

data = pd.read_csv('data/test.csv')

#print(data.columns)
############################## data modeling and scaling##############################
# y = data[['Survived']]
X = data[['PassengerId','Pclass','Sex','Age','SibSp','Parch', 'Fare', 'Embarked']]

X = X.replace(['male','female'], [0,1]) # male == 0 and female == 1
X = X.replace(['C','S','Q'], [1,2,3])

#replacing NaN data with median and mode
X['Age'].fillna(X['Age'].median(),inplace = True)
X['Embarked'].fillna(X['Embarked'].value_counts().idxmax(),inplace = True)
#search if there is any null value
#X.apply(lambda x: sum(x.isnull()))

X['TravleAlone'] = np.where((X['SibSp']+X['Parch'])>0,0,1)
X['FamilySize'] = X['SibSp']+X['Parch']
#modeling title
X['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
X = X.replace(['Mr','Miss','Mrs','Master'], [1,2,3,4])
for index, row in X.iterrows():
    if(type(row['Title'])!=int):
        X['Title'].iloc[index] = 0
   
#print(X['Title'].value_counts())

X['FareBin'] = pd.qcut(X['Fare'], 4)
X['AgeBin'] = pd.cut(X['Age'].astype(int), 5)
# print(X['FareBin'].value_counts())
# print(X['AgeBin'].value_counts())

X['FareBin'] = X.FareBin.astype(str)
X['AgeBin'] = X.AgeBin.astype(str)
agebin_cat = {'(-0.076, 15.2]':1,'(15.2, 30.4]':2,'(30.4, 45.6]':3,'(45.6, 60.8]':4,'(60.8, 76.0]':5}
farebin_cat = {'(-0.001, 7.896]':1,'(7.896, 14.454]':2,'(14.454, 31.472]':3,'(31.472, 512.329]':4}
for index, row in X.iterrows():
    X.set_value(index, 'AgeBin_catg',agebin_cat[row['AgeBin']])
    X.set_value(index, 'FareBin_catg',farebin_cat[row['FareBin']])


#X.drop('SibSp',axis=1,inplace=True)
#X.drop('Parch',axis=1,inplace=True)
# X['Survived'] = y

total = pd.DataFrame([X])
X.to_csv("test_X.csv",header = True)

print("Preprocessing Done!!!!!")
# y.to_csv("main_y.csv",header = True)

##making numpy array
#X_train = np.array(X)
#y = np.array(y)
##X = preprocessing.normalize(X, norm='l2')
#
############################### data plotting##############################
#y_temp_DI = X_train[:,2]
#x_temp_DI = X_train[:,1]
#
#le = ['Pclass', 'Sex']
#plt.figure(figsize=(10, 8))
#plt.title('Attribute Presentation[Scaled]')
#plt.xlabel(le[0])
#plt.ylabel(le[1])
#c=0
#

#for i in y:
#    if(i == 1):
#        plt.plot(x_temp_DI[c],y_temp_DI[c],'g+')
#        c+=1
#    elif(i == 0):
#        plt.plot(x_temp_DI[c],y_temp_DI[c],'r+')
#        c+=1
#  

#
#plt.show()








