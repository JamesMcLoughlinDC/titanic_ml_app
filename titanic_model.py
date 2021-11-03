#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
df["Age"].fillna(df["Age"].median(), inplace = True)
df["Embarked"].fillna("S", inplace = True)
del df["Cabin"]
def getTitle(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else: 
        return "no title in name"
    
titles = set([y for y in df.Name.map(lambda x: getTitle(x))])

def shorter_titles(x):
    title = x["Title"]
    
    if title in ["Capt", "Col", "Major"]:
        return "Officer"
    elif title in ["Don", "Jonkheer", "Lady", "the Countess", "Sir"]:
        return "Royal"
    elif title in ["Mme", "Mrs"]:
        return "Mrs"
    elif title in ["Miss", "Mlle",  "Ms"]:
        return "Miss"
    else:
        return title
df["Title"] = df['Name'].map(lambda x: getTitle(x))

df["Title"] = df.apply( shorter_titles, axis = 1)
df.drop("Name", axis = 1, inplace = True)

df.drop("Ticket", axis = 1, inplace = True)

df.Sex.replace(('male', 'female'), (1,0), inplace =True)
df.Embarked.replace(('S', 'C', 'Q'), (0,1,2), inplace =True)
df.Title.replace(('Mr', 'Mrs', 'Miss', 'Master', 'Dr', 'Rev', 'Officer', 'Royal'), (0,1,2,3,4,5,6,7), inplace =True)

y = df["Survived"]
x = df.drop(["Survived", "PassengerId"], axis = 1)

from sklearn.model_selection import train_test_split


x_train, x_val, y_train, y_val =train_test_split(x,y,test_size = 0.1)

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val)*100, 2)
print(acc_randomforest)

pickle.dump(randomforest, open('titanic_model.sav', 'wb'))


# In[ ]:




