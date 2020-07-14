# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 00:36:55 2020

@author: Rohith
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_csv("E:\\Data Science\\Data_Science_Projects\\Project_2\\Data\\data.csv",encoding='cp1252')
new_data=data.iloc[:,[1,2]]
new_data['Posts']=new_data['Posts'].apply(lambda x:" ".join(x.lower() for x in x.split()))
new_data['Posts']=new_data['Posts'].str.replace('[^\w\s]','')
new_data['Posts']=new_data['Posts'].str.replace('\d+', '')
new_data['Posts']=new_data['Posts'].str.strip()
from textblob import Word
new_data['Posts']=new_data['Posts'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
new_data['Posts']=new_data['Posts'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tv=TfidfVectorizer()

X=new_data.iloc[:,1]
y=new_data.iloc[:,0]

X=tv.fit_transform(new_data.Posts)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from imblearn.over_sampling import SMOTE

sm=SMOTE(random_state=444)

X_train_res, y_train_res = sm.fit_resample(X_train,y_train)
X_train_res.shape
y_train_res.shape
X_test.shape
y_test.shape


model=LogisticRegression()
model.fit(X_train_res,y_train_res)

y_pred=model.predict(X_test)

pickle.dump(tv,open('Transform.pkl', 'wb'))

pickle.dump(model,open('Model.pkl','wb'))  
