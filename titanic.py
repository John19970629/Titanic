
#from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import warnings 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv('Titanic/train.csv')
test=pd.read_csv('Titanic/test.csv')
submit=pd.read_csv('Titanic/gender_submission.csv')

data=train.append(test)
data.reset_index(inplace=True,drop=True)

sns.countplot(data['Survived'])#用直方圖顯示生還狀況
sns.countplot(data['Pclass'],hue=data['Survived'])#觀察艙等跟生存的關聯
sns.countplot(data['Sex'],hue=data['Survived'])#觀察性別跟生存的關聯
sns.countplot(data['Embarked'],hue=data['Survived'])#觀察出發港口跟生存的關聯

g=sns.FacetGrid(data,col='Survived')#觀察年齡層跟生存的關係
g.map(sns.distplot,'Age',kde=False)

g=sns.FacetGrid(data,col='Survived')#觀察票價層跟生存的關聯
g.map(sns.distplot,'Fare',kde=False)

g=sns.FacetGrid(data,col='Survived')#觀察父母數量與生存的關聯
g.map(sns.distplot,'Parch',kde=False)

g=sns.FacetGrid(data,col='Survived')#觀察兄弟姊妹+配偶的數量與生存的關聯
g.map(sns.distplot,'SibSp',kde=False)

data['Family_Size']=data['Parch']+data['SibSp']#將父母+兄弟姊妹+配偶合併成家庭大小
g=sns.FacetGrid(data,col='Survived')
g.map(sns.distplot,'Family_Size',kde=False)


#發現名字的這個欄位有稱謂的資訊(Mr., Miss.) 我們可以利用這些資訊在未來更加提升預測的準確度
data['Title1']=data['Name'].str.split(',',expand=True)[1]
#data['Name'].str.split(',',expand=True).head(3)
data['Title1']=data['Title1'].str.split(',',expand=True)[0]


data['Title2'] = data['Title1'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],['Miss','Mrs','Miss','Mr','Mr','Mrs','Mrs','Mr','Mr','Mr','Mr','Mr','Mr','Mrs'])

#把票號的資訊取出前面英文的部分，因為相同的英文代碼可能代表的是房間的位置，後面的號碼沒有意義所以省略，如果只有號碼的票號就用X來表示
data['Ticket_info'] = data['Ticket'].apply(lambda x : x.replace(".","").replace("/","").strip().split(' ')[0] if not x.isdigit() else 'X')


#用最常出現的"S"填補Embarked中的空值
data['Embarked']=data["Embarked"].fillna('S')

#用Fare的中位數填補Fare中的空值
data['Fare']=data["Fare"].fillna(data['Fare'].mean())

#觀察Cabin的資料後，只取出最前面的英文字母，剩下的用NoCabin來表示
data['Cabin']=data['Cabin'].apply(lambda x:str(x)[0] if not pd.isnull(x) else 'NoCabin')
sns.countplot(data['Cabin'],hue=data['Survived'])

#將類別資料轉為整數
data['Sex']=data['Sex'].astype('category').cat.codes
data['Embarked']=data['Embarked'].astype('category').cat.codes
data['Pclass']=data['Pclass'].astype('category').cat.codes
data['Title1']=data['Title1'].astype('category').cat.codes
data['Title2']=data['Title2'].astype('category').cat.codes
data['Cabin']=data['Cabin'].astype('category').cat.codes
data['Ticket_info']=data['Ticket_info'].astype('category').cat.codes


#隨機森林(推測Age中的空值)
dataAgeNull = data[data["Age"].isnull()]
dataAgeNotNull = data[data["Age"].notnull()]
remove_outlier = dataAgeNotNull[(np.abs(dataAgeNotNull["Fare"]-dataAgeNotNull["Fare"].mean())>(4*dataAgeNotNull["Fare"].std()))|
                      (np.abs(dataAgeNotNull["Family_Size"]-dataAgeNotNull["Family_Size"].mean())>(4*dataAgeNotNull["Family_Size"].std()))                     
                     ]
rfModel_age = RandomForestRegressor(n_estimators=2000,random_state=42)
ageColumns = ['Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title1', 'Title2','Cabin','Ticket_info']
rfModel_age.fit(remove_outlier[ageColumns], remove_outlier["Age"])

ageNullValues = rfModel_age.predict(X= dataAgeNull[ageColumns])
dataAgeNull.loc[:,"Age"] = ageNullValues
data = dataAgeNull.append(dataAgeNotNull)
data.reset_index(inplace=True, drop=True)


#利用隨機森林演算法(Random Forest)來預測存活率
dataTrain = data[pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])
dataTest = data[~pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])

dataTrain = dataTrain[['Survived', 'Age', 'Embarked', 'Fare',  'Pclass', 'Sex', 'Family_Size', 'Title2','Ticket_info','Cabin']]
dataTest = dataTest[['Age', 'Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title2','Ticket_info','Cabin']]

rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=12,
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1) 

rf.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
print("%.4f" % rf.oob_score_)

#submit
rf_res =  rf.predict(dataTest)
submit['Survived'] = rf_res
submit['Survived'] = submit['Survived'].astype(int)
submit.to_csv('submit.csv', index= False)

