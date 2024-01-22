#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
import pandas as pd


# In[97]:


df = pd.read_csv('spam.csv', encoding='ISO-8859-1')


# In[98]:


df.sample(5)


# In[99]:


df.shape


# ## 1. Data Cleaning

# In[100]:


df.info


# In[101]:


df.info()


# In[102]:


df.sample(5)


# In[103]:


df.rename(columns = {'v1' : 'target', 'v2' : 'text'},inplace=True)


# In[104]:


df.sample(5)


# In[105]:


df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],inplace=True)


# In[106]:


df.sample(5)


# In[107]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[108]:


df['target'] = encoder.fit_transform(df['target'])


# In[109]:


df.head()


# In[110]:


df.isnull().sum()


# In[111]:


df.duplicated().sum()


# In[112]:


df = df.drop_duplicates(keep = 'first')


# In[113]:


df.duplicated().sum()


# ## 2. EDA

# In[114]:


df['target'].value_counts()


# In[115]:


import matplotlib.pyplot as plt


# In[116]:


plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct='%.02f')
plt.show()


# In[117]:


# Data is imbalanced


# In[118]:


import nltk


# In[119]:


nltk.download('punkt')


# In[120]:


df['num_alphabets'] = df['text'].apply(len)


# In[121]:


df.head()


# In[122]:


df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))


# In[123]:


df.head()


# In[124]:


df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))


# In[125]:


df.head()


# In[126]:


df[['num_alphabets', 'num_words', 'num_sentences']].describe()


# In[127]:


# For ham
df[df['target']==0][['num_alphabets','num_words','num_sentences']].describe()


# In[128]:


# For spam
df[df['target']==1][['num_alphabets','num_words','num_sentences']].describe()


# In[129]:


import seaborn as sns


# In[130]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_alphabets'])
sns.histplot(df[df['target'] == 1]['num_alphabets'],color = 'red')


# In[131]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color = 'red')


# In[132]:


sns.pairplot(df,hue='target')


# In[133]:


numeric_df = df.select_dtypes(include='number')


# In[134]:


sns.heatmap(numeric_df.corr(),annot=True)


# In[135]:


# Strong multicolinear relation between num_alphabets, num_words and num-sentences
# Since we're keeping one column we would keep num_alphabets as it has highest colinearity with target


# ## 3. Data Preprocessing

# In[ ]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


# In[136]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[137]:


from nltk.corpus import stopwords
nltk.download('stopwords')
import string


# In[140]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[141]:


df['transformed_text'].head()


# In[142]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[143]:


spam_wc = wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))


# In[144]:


plt.imshow(spam_wc)


# In[145]:


ham_wc = wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))


# In[146]:


plt.imshow(ham_wc)


# In[147]:


spam_corpus = []
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[148]:


len(spam_corpus)


# In[149]:


from collections import Counter
spam_common_words = pd.DataFrame(Counter(spam_corpus).most_common(30))
spam_common_words


# In[150]:


spam_common_words.columns = ['words', 'freq']
spam_common_words


# In[151]:


sns.barplot(x='words', y='freq', data=spam_common_words)
plt.xticks(rotation='vertical')
plt.show()


# In[152]:


ham_corpus = []
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[153]:


ham_common_words = pd.DataFrame(Counter(spam_corpus).most_common(30))


# In[154]:


ham_common_words.columns = ['words','freq']


# sns.barplot(x='words', y='freq', data=ham_common_words)
# plt.xticks(rotation='vertical')
# plt.show()

# ## 4. Model Building

# In[190]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()


# In[191]:


X = cv.fit_transform(df['transformed_text']).toarray()


# In[192]:


X.shape


# In[193]:


y = df['target'].values
y


# In[194]:


from sklearn.model_selection import train_test_split


# In[195]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[196]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[197]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[198]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[199]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[200]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[201]:


tfidf = TfidfVectorizer(max_features=3000)
Xt = tfidf.fit_transform(df['transformed_text']).toarray()


# In[202]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# In[203]:


X = np.hstack((X,df['num_alphabets'].values.reshape(-1,1)))


# In[204]:


Xt.shape


# In[205]:


Xt_train, Xt_test, y_train, y_test = train_test_split(Xt,y,test_size=0.2,random_state=2)


# In[206]:


gnb.fit(Xt_train,y_train)
y_pred1 = gnb.predict(Xt_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[207]:


mnb.fit(Xt_train,y_train)
y_pred2 = mnb.predict(Xt_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[208]:


bnb.fit(Xt_train,y_train)
y_pred3 = bnb.predict(Xt_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[189]:


# tfidf mnb


# In[209]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:




