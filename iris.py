#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


# In[33]:


data = pd.read_csv("Iris.csv")
data.head()


# In[34]:


iris_dataset.shape


# In[35]:


print(data.columns)


# In[44]:


from sklearn.model_selection import train_test_split

x=data.iloc[:, :-1]
y=data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# In[45]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = scaler.fit_transform(x)


# In[46]:


from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score

svm_linear = SVC(kernel='linear')

svm_linear.fit(x_train, y_train)


# In[47]:


y_pred = svm_linear.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("précision du modèle svm lineaire : {:.2f}".format(accuracy))


# In[48]:


svm_poly = SVC(kernel='poly', degree=3)

svm_poly.fit(x_train, y_train)

y_pred = svm_poly.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("précision du modèle svm polynomial : {:.2f}".format(accuracy))


# In[49]:


from sklearn.model_selection import cross_val_score

svm_poly = SVC(kernel='poly', degree=3, C=1.0)

scores = cross_val_score(svm_poly, x, y, cv=10)

print("précision moyenne : {:.2f}".format(scores.mean()))
print("écart type : {:.2f}".format(scores.std()))


# In[50]:


svm_poly1 = SVC(kernel='poly', degree=3)

print(svm_poly1)


# In[59]:


# Créer un modèle SVM avec un noyau RBF (fonction de base radiale)
clf = SVC(kernel='rbf', C=1 )

# Entraîner le modèle SVM sur les données d'entraînement
clf.fit(x_train, y_train)

# Faire des prédictions sur les données de test
y_pred = clf.predict(x_test)

# Calculer la précision du modèle
acc = accuracy_score(y_test, y_pred)

# Afficher le score de précision du modèle SVM avec le noyau RBF
print("Score de précision :", acc)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




