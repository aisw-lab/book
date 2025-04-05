#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# # 데이터 불러오기 (학습 데이터, 테스트 데이터)
# 데이터 분석 단계(4.2_농구선수_데이터분석.ipynb)에서 생성한 농구 선수 포지션 예측하기의  
# 학습 데이터 및 테스트 데이터를 로드합니다.

# In[2]:


with open('pkl/basketball_train.pkl', 'rb') as train_data:
    train = pd.read_pickle(train_data)
    
with open('pkl/basketball_test.pkl', 'rb') as test_data:
    test = pd.read_pickle(test_data)


# # 최적의 k 찾기 (교차 검증 - cross validation)

# In[3]:


# import kNN library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# find best k, range from 3 to half of the number of data
max_k_range = train.shape[0] // 2
k_list = []
for i in range(3, max_k_range, 2):
    k_list.append(i)

cross_validation_scores = []
x_train = train[['3P', 'BLK' , 'TRB']]
y_train = train[['Pos']]

# 10-fold cross validation
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train.values.ravel(),
                             cv=10, scoring='accuracy')
    cross_validation_scores.append(scores.mean())

cross_validation_scores


# In[4]:


# visualize accuracy according to k
plt.plot(k_list, cross_validation_scores)
plt.xlabel('the number of k')
plt.ylabel('Accuracy')
plt.show()


# In[5]:


# find best k
cvs = cross_validation_scores
k = k_list[cvs.index(max(cross_validation_scores))]
print("The best number of k : " + str(k) )


# # 2개의 특징으로 예측하기 (3점슛, 블로킹)

# In[6]:


# import libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=k)

# select data features
x_train = train[['3P', 'BLK']]
# select target value
y_train = train[['Pos']]

# setup knn using train data
knn.fit(x_train, y_train.values.ravel())

# select data feature to be used for prediction
x_test = test[['3P', 'BLK']]

# select target value
y_test = test[['Pos']]

# test
pred = knn.predict(x_test)


# In[7]:


# check ground_truth with knn prediction
comparison = pd.DataFrame(
    {'prediction':pred, 'ground_truth':y_test.values.ravel()}) 
comparison


# In[8]:


# check accuracy
print("accuracy : "+ 
          str(accuracy_score(y_test.values.ravel(), pred)) )


# # 3개의 특징으로 예측하기 (3점슛, 블로킹, 리바운드)

# In[9]:


knn = KNeighborsClassifier(n_neighbors=k)

# select data features to be used in train
x_train = train[['3P', 'BLK', 'TRB']]
# select target
y_train = train[['Pos']]

# build knn model
knn.fit(x_train, y_train.values.ravel())

# select features to be used for prediction
x_test = test[['3P', 'BLK', 'TRB']]

# select target
y_test = test[['Pos']]

# test
pred = knn.predict(x_test)


# In[10]:


# check ground_truth with knn prediction
comparison = pd.DataFrame(
    {'prediction':pred, 'ground_truth':y_test.values.ravel()}) 
comparison


# In[11]:


# check accuracy
print("accuracy : " + 
          str( accuracy_score(y_test.values.ravel(), pred)) )

