#!/usr/bin/env python
# coding: utf-8

# **DMDW LAB ASSIGNMENT** 

# **HEART ATTACK ANALYSIS AND PREDICTION BY CHANDAN MOHANTY**

# **SEMESTER- 5TH**
# 
# 
# **SECTION- 'C'**
# 
# 
# 
# **ROLL NO-20CSE012**

# In[2]:


#import all the reqiured lib for the data preporcessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


# import the dataset

df = pd.read_csv(r"C:\Users\91824\Downloads\heart.csv")


# In[6]:


#lets now visualize the dataset
df.head()


# **Inslights of the dataset**

# In[7]:


#This is the stastical value of the dataset
df.describe()


# In[8]:


#information about the dataset
df.info()


# In[9]:


#check for any null values in the dataset
print("Check for null or missing values")
df.isnull().sum()


# So our dataset doesnt have any null or missing values

# **ATTRIBUTES OF DATASET**
# 
# 
# Now lets us explore the attributes of our dataset:
# 
# age - age in years
# 
# sex - sex (1 = male; 0 = female)
# 
# cp - chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 0 = asymptomatic)
# 
# trestbps - resting blood pressure (in mm Hg on admission to the hospital)
# 
# chol - serum cholestoral in mg/dl
# 
# fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
# 
# restecg - resting electrocardiographic results (1 = normal; 2 = having ST-T wave abnormality; 0 = hypertrophy)
# 
# thalach - maximum heart rate achieved
# 
# exang - exercise induced angina (1 = yes; 0 = no)
# 
# oldpeak - ST depression induced by exercise relative to rest
# 
# slope - the slope of the peak exercise ST segment (2 = upsloping; 1 = flat; 0 = downsloping)
# 
# ca - number of major vessels (0-3) colored by flourosopy
# 
# thal - 2 = normal; 1 = fixed defect; 3 = reversable defect
# 
# num - the predicted attribute - diagnosis of heart disease 

# In[13]:


print("This the shape of dataset: " ,df.shape)


# In[19]:


#Now let us find how many unique elements are there in each attribute
uniq_number = []
for i in df.columns:
    x = df[i].value_counts().count()
    uniq_number.append(x)

pd.DataFrame(uniq_number, index = df.columns, columns=['Total Unique Values'])


# According to the result from the unique value dataframe
# 
# 
# 
# We determined the variables with few unique values as categorical variables, and the variables with high unique values as numeric variables.
# 
# 
# 
# **Numeric Variables: “age”, “trtbps”, “chol”, “thalach” and “oldpeak ”**
# 
# 
# 
# 
# 
# 
# **Categorical Variables: "sex", "cp", "fbs", "rest_ecg", "exng", "slope", "caa", "thall", "target"**

# In[20]:


# this is out the targeted attribute
df['output'].unique()


# In[21]:


#Lets check the correlation between the targeted attribute and other feature attribute
print(df.corr()['output'].abs().sort_values(ascending=False))


# **EDA - EXPLOTARY DATA ANALYSIS**
# 
# 
# **NOW LETS VISUALISE THE DATASET**

# In[22]:


sns.countplot(x=df['output'])


# In[26]:


#Box plot representation of the feature attributre

f, axes = plt.subplots(4,1,figsize = (20,20))

sns.boxplot(df['age'], ax=axes[0])
sns.boxplot(df['caa'],ax=axes[1])
sns.boxplot(df['thall'], ax=axes[2])
sns.boxplot(df['cp'],ax=axes[3])


# In[33]:


# Barplot Representation of Targeted Attribute vs Feature Attribute

sns.barplot(x=df["sex"],y=df["output"])


# In[34]:



sns.barplot(x=df["cp"],y=df["output"])


# In[35]:


sns.barplot(x=df["restecg"],y=df["output"])


# In[38]:


sns.barplot(x=df["exng"],y=df["output"])


# In[40]:


sns.barplot(x=df["fbs"],y=df["output"])


# In[43]:


sns.barplot(x=df["slp"],y=df["output"])


# In[45]:


sns.barplot(x=df["caa"],y=df["output"])


# In[47]:


sns.barplot(x=df["thall"],y=df["output"])


# **TRAIN AND TEST SPLIT OF DATASET**

# In[48]:


from sklearn.model_selection import train_test_split   #import the fuction to split the data
features_df = df.drop("output",axis=1)  # select all the attributes except the targeted attribute
target= df["output"]

X_train, X_test, Y_train, Y_test = train_test_split(features_df, target, test_size=0.20,random_state=0)


# In[50]:


# Now lets visualize the training dataset and testing dataset
print("SHAPE OF FEATURE TRAINING DATASET:",X_train.shape)
print("SHAPE OF FEATURE TESTING DATASET : ",X_test.shape)
print("SHAPE OF TARGETED TRAINING DATASET : ",Y_train.shape)
print("SHAPE OF TARGETED TESTING DATASET : ",Y_test.shape)


# **KNN ALGO IMPLIMENTATION**
# 
# 
# 
# **KNN - K-NEAREST NEIGHBOUR**

# In[51]:


from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors = 7) # we chose 7 as because its a safe number


# In[53]:


from sklearn.metrics import accuracy_score
knn.fit(X_train,Y_train)  #this fit function is used to fit the dataset into the model and train it
knn_pred = knn.predict(X_test)
accuracy_knn = round(accuracy_score(knn_pred,Y_test)*100,2)
print("ACCURACY OF KNN: "+str(accuracy_knn)+" %")


# **DECISION TREE ALGO IMPLIMENTATION**

# In[54]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()


# In[55]:


dt.fit(X_train,Y_train)
dt_pred = dt.predict(X_test)
accuracy_dt = round(accuracy_score(dt_pred,Y_test)*100,2)
print("THE ACCURACY OF DECISION TREE: "+str(accuracy_dt)+" %")


# **NAVIE BAYES CLASSIFER IMPLIMENTATION**

# In[56]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()


# In[58]:


nb.fit(X_train,Y_train)
nb_pred = nb.predict(X_test)
accuracy_nb = round(accuracy_score(nb_pred,Y_test)*100,2)
print("THE ACCURACY OF NAIVE BAYES: "+str(accuracy_nb)+" %")


# In[60]:


algo=['NAVIE BAYES', 'DESCISION TREE', 'K-NEAREST NEIGHBOUR']
score=[accuracy_nb,accuracy_dt,accuracy_knn]


# In[62]:


plt.xlabel("Algorithms")
plt.ylabel("ACCURACY")
sns.barplot(x=algo,y=score)


# **Conclusion:**

# **FROM THE ABOVE EXPERIMENT AND ANALYSIS OF THE DATASET OVER DIFFERENT ALGORITHMS WE FOUND THAT ,THE BEST ALGORITHM FOR OUR DATASET IS NAVIE BAYES**

# In[ ]:




