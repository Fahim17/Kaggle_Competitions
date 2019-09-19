import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier

RanFor_Clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
SVC_Clf = SVC(gamma='auto')
GrdBost_Clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
QuaDisAna_Clf = QuadraticDiscriminantAnalysis()
AdaBost_Clf = AdaBoostClassifier(n_estimators=100, random_state=0)	


data = pd.read_csv('main_X.csv')
data_testX = pd.read_csv('test_X.csv')
y = data[['Survived']]
X = data[['Pclass','Sex','Age','SibSp','Parch', 'Embarked','TravleAlone','FamilySize','Title','AgeBin_catg','FareBin_catg']]
testX = data_testX[['Pclass','Sex','Age','SibSp','Parch', 'Embarked','TravleAlone','FamilySize','Title','AgeBin_catg','FareBin_catg']]

X = np.array(X)
y = np.array(y)
testX = np.array(testX)
############################## ################ ##############################
scores = {}
scores["RanFor_Clf"] = cross_val_score(RanFor_Clf, X, y, cv=10, scoring = 'accuracy').mean()
# print("RanFor_Clf score = ",scores.mean())

scores["SVC_Clf"] = cross_val_score(SVC_Clf, X, y, cv=10, scoring = 'accuracy').mean()
# print("SVC_Clf score = ",scores.mean())

scores["GrdBost_Clf"] = cross_val_score(GrdBost_Clf, X, y, cv=10, scoring = 'accuracy').mean()
# print("GrdBost_Clf score = ",scores.mean())

scores["QuaDisAna_Clf"] = cross_val_score(QuaDisAna_Clf, X, y, cv=10, scoring = 'accuracy').mean()
# print("QuaDisAna_Clf score = ",scores.mean())

scores["AdaBost_Clf"] = cross_val_score(AdaBost_Clf, X, y, cv=10, scoring = 'accuracy').mean()
# print("AdaBost_Clf score = ",scores.mean())

print(scores)

############################## ################ ##############################
# predt = pd.DataFrame()
# predt['PassengerId'] = data_testX['PassengerId']

# RanFor_Clf.fit(X,y)
# predt['RanFor'] = RanFor_Clf.predict(testX)

# SVC_Clf.fit(X,y)
# predt['SVC'] = SVC_Clf.predict(testX)

# GrdBost_Clf.fit(X,y)
# predt['GrdBost'] = GrdBost_Clf.predict(testX)

# QuaDisAna_Clf.fit(X,y)
# predt['QuaDisAna'] = QuaDisAna_Clf.predict(testX)

# AdaBost_Clf.fit(X,y)
# predt['AdaBost'] = AdaBost_Clf.predict(testX)

# for index, row in predt.iterrows():
# 	tot = row['RanFor']+row['SVC']+row['GrdBost']+row['QuaDisAna']+row['AdaBost']

# 	if tot < 3:
# 		predt.set_value(index, 'Survived',0)
# 	else:
# 		# predt.set_value(index, 'final',1)
# 		predt.set_value(index, 'Survived',1)


# # print(predt['final'])
# predt[['PassengerId','Survived']].to_csv("submission.csv",header = True, index = False)
# print("All done!!")