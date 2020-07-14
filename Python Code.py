# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:57:48 2020

@author: Rohith
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_csv("E:\\Data Science\\Project_2\\Data\\data.csv",encoding='cp1252')

df_a=data.loc[data.Type=='A']
df_b=data.loc[data.Type=='B']
df_c=data.loc[data.Type=='C']
df_d=data.loc[data.Type=='D']



new_data=data.iloc[:,[1,2]]

X=data.iloc[:,2]
y=data.iloc[:,1]

#rows and columns returns (rows, columns)
new_data.shape

#rows and columns returns (rows, columns)
new_data.head()

#returns the last x number of rows when tail(num). Without a number it returns 5
new_data.tail()

#returns an object with all of the column headers 
new_data.columns

#basic information on all columns 
new_data.info()

#shows what type the data was read in as (float, int, string, bool, etc.)
new_data.dtypes

#shows which values are null
new_data.isnull().sum()

#shows the counts for those unique values 
new_data.Type.value_counts()

new_data.describe()

# Plots

#histogram 

new_data.Type.hist()
new_data.Type.hist(bins=10)
plt.xlabel('Type')
plt.ylabel('Count')

#bar chart of types 
data.Type.value_counts().plot(kind='bar')
plt.xlabel('Type')
plt.ylabel('Count')

##To find the number of words
a=new_data['Posts'].apply(lambda x: len(str(x).split(" ")))
a.head()

### To find the average word length
def avg_word(sentence):
    words=sentence.split(" ")
    return(sum(len(word) for word in words)/len(words))
    
b=new_data['Posts'].apply(lambda x: avg_word(x))

b.head()

## To find the length of stopwords
from nltk.corpus import stopwords
stop=stopwords.words("english")
c=new_data['Posts'].apply(lambda x: len([x for x in x.split() if x in stop]))

c.head()

### To find the number of special characters
d=new_data['Posts'].apply(lambda x:sum(x.count(s) for s in '#!*$&@%?+-^.,'))
d.head()
          
## To find the count of  numerics
e=new_data['Posts'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
e.head()

## number of uppercase words
f=new_data['Posts'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
f.head()


######################## NLP:Text processing  ########################## 

##lower case
new_data['Posts']=new_data['Posts'].apply(lambda x:" ".join(x.lower() for x in x.split()))


## removing punctuations
new_data['Posts']=new_data['Posts'].str.replace('[^\w\s]','')

##removal of digits
new_data['Posts']=new_data['Posts'].str.replace('\d+', '')

### removing all stopwords(english)....###
from nltk.corpus import stopwords

stop_words=stopwords.words('english')

new_data['Posts']=new_data['Posts'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))


##removal of white spaces 
new_data['Posts']=new_data['Posts'].str.strip()


####Lemmatization
from textblob import Word
new_data['Posts']=new_data['Posts'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


##to remove links in text data
new_data['Posts']=new_data['Posts'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)


################# bigrams for Posts ###############################################
from textblob import TextBlob
import collections
from collections import Counter
import nltk

for i in new_data['Posts'][0:10]:
    grams=TextBlob(i).ngrams(2)
    print(grams)
    
counts = collections.Counter()
for i in new_data['Posts']:
    words1 =i.split()
    counts.update(nltk.bigrams(words1))
    
common_bigrams = counts.most_common(10)
common_bigrams
############################################################################################

########...finding Correlation in the data....###
corrmat = new_data.corr()
print(corrmat)


## ....Finding most common occuring words in Corpus...##
posts_str=" ".join(new_data.Posts)
text=posts_str.split()

from collections import Counter
counter= Counter(text)
top_100= counter.most_common(100)
print(top_100)

####################################################################################################
 ###########Model_Building####################################
 ###Logistic_Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tv=TfidfVectorizer()

X=new_data.iloc[:,1]
y=new_data.iloc[:,0]

X=tv.fit_transform(new_data.Posts)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

model=LogisticRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

##########################################################################################################################

####Decision Trees:

from sklearn.tree import DecisionTreeClassifier

model1=DecisionTreeClassifier(criterion='entropy')
model1.fit(X_train,y_train)
y_pred1=model1.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(classification_report(y_test,y_pred1))

##############################################################################################################################

#####Random Forest

from sklearn.ensemble import RandomForestClassifier

model2=RandomForestClassifier(n_estimators=30)
model2.fit(X_train,y_train)

y_pred2=model2.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))

#########################################################################################################################################

###########Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB

model3=MultinomialNB()
model3.fit(X_train,y_train)

y_pred3=model3.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(classification_report(y_test,y_pred3))

#########################################################################################################################################################

##################### SVM

from sklearn.svm import SVC

model4=SVC()
model4.fit(X_train,y_train)

y_pred4=model4.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred4))
print(confusion_matrix(y_test,y_pred4))
print(classification_report(y_test,y_pred4))

###############################################################################################################################################################

###############Neural_Ntworks

from sklearn.neural_network import MLPClassifier

model5=MLPClassifier(hidden_layer_sizes=(5,5))

model5.fit(X_train,y_train)

y_pred5=model5.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred5))
print(confusion_matrix(y_test,y_pred5))
print(classification_report(y_test,y_pred5))

##############################################################################################################################################################

#######################Bagging Classifier

from sklearn.ensemble import BaggingClassifier

model6=BaggingClassifier(DecisionTreeClassifier(criterion='entropy'))
model6.fit(X_train,y_train)

y_pred6=model6.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred6))
print(confusion_matrix(y_test,y_pred6))
print(classification_report(y_test,y_pred6))


##################################################################################################################################################################

#########Applying SMOTE to overcome class imbalance problem####################

from imblearn.over_sampling import SMOTE

sm=SMOTE(random_state=444)

X_train_res, y_train_res = sm.fit_resample(X_train,y_train)
X_train_res.shape
y_train_res.shape
X_test.shape
y_test.shape


model7=LogisticRegression()
model7.fit(X_train_res,y_train_res)

y_pred7=model7.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred7))
print(classification_report(y_test,y_pred7))
print(confusion_matrix(y_test,y_pred7))

########################################################################################


from sklearn.tree import DecisionTreeClassifier

model8=DecisionTreeClassifier(criterion='entropy')
model8.fit(X_train_res,y_train_res)
y_pred8=model8.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred8))
print(confusion_matrix(y_test,y_pred8))
print(classification_report(y_test,y_pred8))

##############################################################################################

from sklearn.ensemble import RandomForestClassifier

model9=RandomForestClassifier(n_estimators=100)
model9.fit(X_train_res,y_train_res)

y_pred9=model9.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred9))
print(confusion_matrix(y_test,y_pred9))
print(classification_report(y_test,y_pred9))

###############################################################################################

from sklearn.naive_bayes import MultinomialNB

model10=MultinomialNB()
model10.fit(X_train_res,y_train_res)

y_pred10=model10.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred10))
print(confusion_matrix(y_test,y_pred10))
print(classification_report(y_test,y_pred10))

################################################################################################


from sklearn.svm import SVC

model11=SVC()
model11.fit(X_train_res,y_train_res)

y_pred11=model11.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred11))
print(confusion_matrix(y_test,y_pred11))
print(classification_report(y_test,y_pred11))

########################################################################################################

from sklearn.neural_network import MLPClassifier

model12=MLPClassifier(hidden_layer_sizes=(5,5))

model12.fit(X_train_res,y_train_res)


y_pred12=model12.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred12))
print(confusion_matrix(y_test,y_pred12))
print(classification_report(y_test,y_pred12))


###################################################################################################

from sklearn.ensemble import BaggingClassifier

model13=BaggingClassifier(DecisionTreeClassifier(criterion='entropy'))
model13.fit(X_train_res,y_train_res)

y_pred13=model13.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred13))
print(confusion_matrix(y_test,y_pred13))
print(classification_report(y_test,y_pred13))


###############################################################################################

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

model14 = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001,C=0.55, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=42, max_iter=1000)


model14.fit(X_train_res,y_train_res)

y_pred14=model14.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred14))
print(confusion_matrix(y_test,y_pred14))
print(classification_report(y_test,y_pred14))

#####################################################################################################
































