import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('white')
import nltk
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

# Loading dataset

data_post= pd.read_csv('E:\\excelr\\excelr\\my project NLP\\pjt 2\\data set details\\NAOP_new\\NAOP.csv', encoding='cp1252',error_bad_lines=False) 

data_post.head(10)
data_post.columns
data_post.shape
data_post.info()
data_post.describe()

#################################Exploratory Data Analysis######################################

category=data_post.groupby(['Type']).count() # counting different category 
category
###finding Number of unique values
data_post.nunique()
##Checking Null values
data_post.isnull().sum()
# Dataset in Graphs
new_data=data_post.iloc[:,[1,2]]
X=new_data.iloc[:,1]
y=new_data.iloc[:,0]
new_data.Type.hist()
new_data.Type.hist(bins=10)
plt.xlabel('Type')
plt.ylabel('Count')
#bar chart of types 
new_data.Type.value_counts().plot(kind='pie')
plt.xlabel('Type')
plt.ylabel('Count')

##considering Type and posts Columns
posts=data_post.iloc[:,[1,2]] # Selecting Type and post columns
posts.shape
posts.describe()
posts.head(10)
category_posts=posts.groupby(['Type']).count() # counting different Type 
category_posts
posts.dtypes

# Seperating Types
df_a = posts.loc[data_post.Type=='A']
df_b = posts.loc[data_post.Type=='B']
df_c = posts.loc[data_post.Type=='C']
df_d = posts.loc[data_post.Type=='D']
print(df_a, df_b, df_c, df_d)

category_A=pd.DataFrame(df_a.Posts)
category_B=pd.DataFrame(df_b.Posts)
category_C=pd.DataFrame(df_c.Posts)
category_D=pd.DataFrame(df_d.Posts)

############### Basic feature extraction of text data #################
# number of characters
posts['Char_count']=posts['Posts'].str.len()
print(posts['Char_count'])
posts['word_count']=posts['Posts'].apply(lambda x: len(str(x).split(" ")))
print(posts['word_count'])

avg_count=posts['word_count'].groupby(posts['Type']).mean()
avg_count1=pd.DataFrame(avg_count)
avg_count1
avg_count1['category']=['df_a','df_b','df_c','df_d']
avg_count1.head()

#avgerage word_count plot

#plt.figure(figsize=(10,5))
#plt.axis("equal")
#plt.bar("word_count,labels=category")
sns.barplot(data=avg_count1, x="category", y="word_count")
#plt.show()

# Remove ||| from dataset
import re

category_A['Posts'] =category_A['Posts'].apply(lambda x:(re.sub("[]|||[]", " ", x)))
category_A['Posts'].head()
category_B['Posts'] =category_B['Posts'].apply(lambda x:(re.sub("[]|||[]", " ", x)))
category_B['Posts'].head()
category_C['Posts'] =category_C['Posts'].apply(lambda x:(re.sub("[]|||[]", " ", x)))
category_C['Posts'].head()
category_D['Posts'] =category_D['Posts'].apply(lambda x:(re.sub("[]|||[]", " ", x)))
category_D['Posts'].head()

#removing punctuations, WhiteSpaces & to lower
# Category_A
category_A['Posts']=category_A['Posts'].apply(lambda x:" ".join(x.lower() for x in x.split()))
category_A['Posts']=category_A['Posts'].str.strip()
category_A['Posts']=category_A['Posts'].apply(lambda x:(re.sub('[^A-Za-z]', ' ', x)))
category_A['Posts'].head()
# Category_B
category_B['Posts']=category_B['Posts'].apply(lambda x:" ".join(x.lower() for x in x.split()))
category_B['Posts']=category_B['Posts'].str.strip()
category_B['Posts']=category_B['Posts'].apply(lambda x:(re.sub('[^A-Za-z]', ' ', x)))
category_B['Posts'].head()
# Category_C
category_C['Posts']=category_C['Posts'].apply(lambda x:" ".join(x.lower() for x in x.split()))
category_C['Posts']=category_C['Posts'].str.strip()
category_C['Posts']=category_C['Posts'].apply(lambda x:(re.sub('[^A-Za-z]', ' ', x)))
category_C['Posts'].head()
# Category_D
category_D['Posts']=category_D['Posts'].apply(lambda x:" ".join(x.lower() for x in x.split()))
category_D['Posts']=category_D['Posts'].str.strip()
category_D['Posts']=category_D['Posts'].apply(lambda x:(re.sub('[^A-Za-z]', ' ', x)))
category_D['Posts'].head()

#removing stopwords
from nltk.corpus import stopwords
stop=stopwords.words('english')

# Removing unwanted words from frequent list.
stop.append('intp')
stop.append('ínfp')
stop.append('infj')
stop.append('intj')
stop.append('ísfj')
stop.append('isfp')
stop.append('enfp')
stop.append('entp')
stop.append('istp')
stop.append('esfj')
stop.append('estp')
stop.append('esfp')
stop.append('estj')
stop.append('estps')
stop.append('qsxhcwe')
stop.append('http')
stop.append('www')
stop.append('com')
stop.append('fhigbolffgw')
stop.append('plaaikvhvzs')
stop.append('sc')
stop.append('fl')

# Category_A
category_A['Posts']=category_A['Posts'].apply(lambda x:" ".join(w for w in x.split() if w not in stop))
category_A['Posts'].head()

# Category_B
category_B['Posts']=category_B['Posts'].apply(lambda x:" ".join(w for w in x.split() if w not in stop))
category_B['Posts'].head()
# Category_C
category_C['Posts']=category_C['Posts'].apply(lambda x:" ".join(w for w in x.split() if w not in stop))
category_C['Posts'].head()
# Category_D
category_D['Posts']=category_D['Posts'].apply(lambda x:" ".join(w for w in x.split() if w not in stop))
category_D['Posts'].head()

# Lemmatization
category_A['Posts'] = category_A['Posts'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
category_A['Posts'].head()
# Category_B
category_B['Posts'] = category_B['Posts'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
category_B['Posts'].head()
# Category_C
category_C['Posts'] = category_C['Posts'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
category_C['Posts'].head()
# Category_D
category_D['Posts'] = category_D['Posts'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
category_D['Posts'].head()

######## Removing Individual chatacter from datasets ############

pattern = r"((?<=^)|(?<= )).((?=$)|(?= ))"

# (?<=^) & (?<= ) ->are look-behinds for start of string and space, respectively,
# Match either of these conditions using | (or).
# . matches any single character
# (?=$)|(?= ) -> is similar to the first bullet point, except it's a look-ahead
# for either the end of the string or a space.
# Finally call re.sub("\s+", " ", my_string) to condense multiple spaces with a single space.

category_A['Posts'] =category_A['Posts'].apply(lambda x:(re.sub(pattern, '',x).strip()))
category_A['Posts'].head()
category_B['Posts'] =category_B['Posts'].apply(lambda x:(re.sub(pattern, '',x).strip()))
category_B['Posts'].head()
category_C['Posts'] =category_C['Posts'].apply(lambda x:(re.sub(pattern, '',x).strip()))
category_C['Posts'].head()
category_D['Posts'] =category_D['Posts'].apply(lambda x:(re.sub(pattern, '',x).strip()))
category_D['Posts'].head()

############## Word Cloud of All the Four Categories #################

category_text_A = ' '.join(category_A['Posts'])
Q_wordcloud=WordCloud(
                    background_color='white',
                    width=2000,
                    height=2000
                   ).generate(category_text_A)
fig = plt.figure(figsize = (10, 10))
plt.axis('on')
plt.imshow(Q_wordcloud)

# Category_B
category_text_B = ' '.join(category_B['Posts'])
Q_wordcloud_B=WordCloud(
                    background_color='white',
                    width=2000,
                    height=2000
                   ).generate(category_text_B)
fig = plt.figure(figsize = (10, 10))
plt.axis('on')
plt.imshow(Q_wordcloud_B)

# Category_C
category_text_C = ' '.join(category_C['Posts'])
Q_wordcloud_C=WordCloud(
                    background_color='white',
                    width=2000,
                    height=2000
                   ).generate(category_text_C)
fig = plt.figure(figsize = (10, 10))
plt.axis('on')
plt.imshow(Q_wordcloud_C)

# Category_D
category_text_D = ' '.join(category_D['Posts'])
Q_wordcloud_D=WordCloud(
                    background_color='white',
                    width=2000,
                    height=2000
                   ).generate(category_text_D)
fig = plt.figure(figsize = (10, 10))
plt.axis('on')
plt.imshow(Q_wordcloud_D)

###top 20 most frequent repeated words in all categories (Posts)
freq_A = pd.Series(' '.join(category_A['Posts']).split()).value_counts()[0:20]
freq_A
freq_B = pd.Series(' '.join(category_B['Posts']).split()).value_counts()[:20]
freq_B
freq_C = pd.Series(' '.join(category_C['Posts']).split()).value_counts()[:20]
freq_C
freq_D = pd.Series(' '.join(category_D['Posts']).split()).value_counts()[:20]
freq_D

################# bigrams for Categories #####################
import collections
from collections import Counter
# Category_A
counts_A = collections.Counter()
for i in category_A['Posts']:
    words_A = word_tokenize(i)
    counts_A.update(nltk.bigrams(words_A))    
Bigram_category_A = counts_A.most_common(10)
Bigram_category_A

# Category_B
counts_B = collections.Counter()
for i in category_B['Posts']:
    words_B = word_tokenize(i)
    counts_B.update(nltk.bigrams(words_B))    
Bigram_category_B = counts_B.most_common(10)
Bigram_category_B
# Category_C
counts_C = collections.Counter()
for i in category_C['Posts']:
    words_C = word_tokenize(i)
    counts_C.update(nltk.bigrams(words_C))    
Bigram_category_C = counts_C.most_common(10)
Bigram_category_C
# Category_D
counts_D = collections.Counter()
for i in category_D['Posts']:
    words_D = word_tokenize(i)
    counts_D.update(nltk.bigrams(words_D))    
Bigram_category_D = counts_D.most_common(10)
Bigram_category_D

############ TFIDF matrix ################
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
category_A_TFIDF=tfidf.fit_transform(category_A['Posts'])
print(category_A_TFIDF)
# Category_B
category_B_TFIDF=tfidf.fit_transform(category_B['Posts'])
print(category_B_TFIDF)
# Category_C
category_C_TFIDF=tfidf.fit_transform(category_C['Posts'])
print(category_C_TFIDF)
# Category_D
category_D_TFIDF=tfidf.fit_transform(category_D['Posts'])
print(category_D_TFIDF)

################## Sentiment Analysis #####################3
from textblob import TextBlob
category_A['polarity'] = category_A['Posts'].apply(lambda x: TextBlob(x).sentiment[0])
category_A[['Posts','polarity']].head(5)
# Category_B
category_B['polarity'] = category_B['Posts'].apply(lambda x: TextBlob(x).sentiment[0] )
category_B[['Posts','polarity']].head(5)
# Category_C
category_C['polarity'] = category_C['Posts'].apply(lambda x: TextBlob(x).sentiment[0] )
category_C[['Posts','polarity']].head(5)
# Category_D
category_D['polarity'] = category_D['Posts'].apply(lambda x: TextBlob(x).sentiment[0] )
category_D[['Posts','polarity']].head(5)

# Displaying top 5 positive posts of Category_A
category_A[category_A.polarity>0].head(5)
# # Displaying top 5 negative posts of Category_A
category_A[category_A.polarity<0].head(5)
# # Displaying top 5 Neutral posts of Category_A
category_A[category_A.polarity==0].head(5)

# Category_B
# Displaying top 5 positive posts of Category_B
category_B[category_B.polarity>0].head(5)
#  Displaying top 5 negative posts of Category_B
category_B[category_B.polarity<0].head(5)
#  Displaying top 5 Neutral posts of Category_B
category_B[category_B.polarity==0].head(5)

# Category_C
# Displaying top 5 positive posts of Category_C
category_C[category_C.polarity>0].head(5)
#  Displaying top 5 negative posts of Category_C
category_C[category_C.polarity<0].head(5)
#  Displaying top 5 Neutral posts of Category_C
category_C[category_C.polarity==0].head(5)

# Category_D
# Displaying top 5 positive posts of Category_D
category_D[category_D.polarity>0].head(5)
#  Displaying top 5 negative posts of Category_D
category_D[category_D.polarity<0].head(5)
#  Displaying top 5 Neutral posts of Category_C
category_D[category_D.polarity==0].head(5)

# ======= The distribution of Categories polarity score =======
    
sns.set()
plt.hist(x='polarity', data=category_A, bins=20);
plt.xlabel('polarity of category_A');
plt.ylabel('count'); 
plt.figsize=(10, 16)

# Category_B
sns.set()
plt.hist(x='polarity', data=category_B, bins=20);
plt.xlabel('polarity of category_B');
plt.ylabel('count'); 
plt.figsize=(10, 16)

# Category_C
sns.set()
plt.hist(x='polarity', data=category_C, bins=20);
plt.xlabel('polarity of category_C');
plt.ylabel('count'); 
plt.figsize=(10, 16)

# Category_D
sns.set()
plt.hist(x='polarity', data=category_D, bins=20);
plt.xlabel('polarity of category_D');
plt.ylabel('count'); 
plt.figsize=(10, 16)

 def sent_type(text): 
    for i in (text):
        if i>0:
            print('positive')
        elif i==0:
            print('netural')
        else:
            print('negative') 
            
sent_type(category_A['polarity'])
sent_type(category_B['polarity'])
sent_type(category_C['polarity'])
sent_type(category_D['polarity'])
