#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None


# In[2]:


df = pd.read_csv('Train.csv')


# In[3]:


df.head()


# In[4]:


# Data Cleaning


# In[12]:


#write the columns names in small letter and replace spaces with under score
df.columns=df.columns.str.lower().str.replace(' ','_')


# In[6]:


for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print()


# In[7]:


df.dtypes


# In[8]:


strings =list(df.dtypes[df.dtypes=='object'].index)
strings


# In[9]:


for col in strings:
    df[col] = df[col].str.lower().str.replace(' ','_')


# In[10]:


df.head()


# # EDA

# In[13]:


df.mode_of_shipment.value_counts()


# In[14]:


df.reached_on_time_y_n.value_counts(normalize = True)


# In[15]:


# plot between mode_of_shipment and reached_on_time_y_n
sns.countplot(y=df['mode_of_shipment'], hue=df['reached_on_time_y_n'])


# In[16]:


df.warehouse_block.value_counts()


# In[17]:


# plot between warehouse_block and reached_on_time_y_n
sns.countplot(y=df['warehouse_block'], hue=df['reached_on_time_y_n'])


# In[18]:


df.customer_care_calls.value_counts()


# In[19]:


# plot between customer_care_calls and reached_on_time_y_n
sns.countplot(y=df['customer_care_calls'], hue=df['reached_on_time_y_n'])


# In[20]:


df.customer_rating.value_counts()


# In[21]:


# plot between customer_rating and reached_on_time_y_n
sns.countplot(y=df['customer_rating'], hue=df['reached_on_time_y_n'])


# In[22]:


df.corr(method ='pearson')


# In[23]:


#Checking for corrolation
corrolation_matrix=df.corr(method ='pearson')

sns.heatmap(corrolation_matrix,annot=True)
plt.title('Corrolation reached_on_time_y_n')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()


# # Setting up Validation Frame Work

# In[24]:


# perfom train/validation/test using sklearn
from sklearn.model_selection import train_test_split


# In[25]:


# divide the data into train,validation,test
df_full_train, df_test =train_test_split(df, test_size = 0.2, random_state=42 )


# In[26]:


len(df_full_train), len(df_test)


# In[27]:


df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)


# In[28]:


len(df_train), len(df_val), len(df_test)


# In[29]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[30]:


y_train = df_train.reached_on_time_y_n.values
y_val = df_val.reached_on_time_y_n.values
y_test = df_test.reached_on_time_y_n.values


# In[31]:


del df_train['reached_on_time_y_n']
del df_test['reached_on_time_y_n']
del df_val['reached_on_time_y_n']


# # Mutual Informaton

# In[32]:


# mutual information : concept from information theory that tells us how much we can learn about one variable if we know
#the value of another
from sklearn.metrics import mutual_info_score


# In[33]:


mutual_info_score(df_full_train.reached_on_time_y_n, df_full_train.product_importance)


# In[34]:


# apply mutual info to the whole data
def mutual_info_reached_on_time_y_n_score(series):
    return mutual_info_score(series, df_full_train.reached_on_time_y_n)


# In[35]:


mi = df_full_train.apply(mutual_info_reached_on_time_y_n_score)


# In[36]:


# sort by des
mi.sort_values(ascending = False)


# # One hot Encoding

# In[37]:


# use one hot encoding for categorical variables
from sklearn.feature_extraction import DictVectorizer


# In[38]:


train_dicts = df_train.to_dict(orient='records')


# In[39]:


train_dicts[0]


# In[40]:


dv = DictVectorizer(sparse=False)


# In[41]:


X_train = dv.fit_transform(train_dicts)


# In[42]:


val_dicts = df_val.to_dict(orient='records')


# In[43]:


X_val = dv.transform(val_dicts)


# # Logistic Regression

# In[44]:


from sklearn.linear_model import LogisticRegression


# In[45]:


model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)


# In[46]:


model.fit(X_train, y_train)


# In[47]:


model.intercept_[0]


# In[48]:


# this is our w(weight)
model.coef_[0].round(3)


# In[49]:


# hard prediction predict 0 and 1
model.predict(X_train)


# In[50]:


# soft probability predict likelyhood of churning
model.predict_proba(X_train)


# In[51]:


# using it on valitaton dataset
y_pred =model.predict_proba(X_val)[:,1]


# In[52]:


reached_on_time_y_n_decision = (y_pred >= 0.5)


# In[53]:


# check the accuracy of the model
accuracy = (y_val == reached_on_time_y_n_decision).mean()


# In[54]:


print(round(accuracy,2))


# In[55]:


df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] =reached_on_time_y_n_decision.astype(int)
df_pred['actual'] = y_val


# In[56]:


df_pred['correct'] = df_pred.prediction == df_pred.actual


# In[57]:


df_pred


# In[58]:


(y_val == reached_on_time_y_n_decision).sum()


# In[59]:


len(X_val)


# In[60]:


1420/2200


# In[61]:


thresholds = np.linspace(0, 1,21)

scores= []
for t in thresholds:
    reached_on_time_y_n_decision =(y_pred >=t)
    score = (y_val == reached_on_time_y_n_decision).mean()
    print('%.2f%.3f'%(t,score))
    scores.append(score)


# In[62]:


plt.plot(thresholds, scores)


# In[63]:


from sklearn.metrics import accuracy_score


# In[64]:


accuracy_score(y_val,y_pred >=0.65)


# In[65]:


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[66]:


alpha_values = [0, 0.01, 0.1, 1, 10]


# In[67]:


rmse_score = {}


# In[68]:


for alpha in alpha_values:
    model = Ridge(alpha=alpha,solver='sag', random_state=42)
    model.fit(X_train, y_train)


# In[69]:


y_pred = model.predict(X_val)


# In[70]:


rmse =round(sqrt(mean_squared_error(y_val,y_pred)),3)
rmse_score[alpha]= rmse


# In[71]:


for alpha, rmse in rmse_score.items():
    print(f'Alpha={alpha}: RMSE ={rmse}')


# In[72]:


from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# In[73]:


actual_positive=(y_val==1)

actual_negative=(y_val==0)


# In[74]:


len(y_val)


# In[75]:


(y_val == reached_on_time_y_n_decision).mean()


# In[76]:


(y_val == reached_on_time_y_n_decision).sum()


# In[77]:


t=0.5
predict_positive = (y_pred >=t)
predict_negative = (y_pred < t)


# In[78]:


tp = (actual_positive & predict_positive).sum()
tn = (predict_negative & actual_negative).sum()


# In[79]:


fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()


# In[80]:


tp, tn, fp, fn


# In[81]:


p = tp / (tp + fp)
p


# In[82]:


r = tp / (tp + fn)
r


# In[83]:


scores =  []
thresholds = np.arange(0.0, 1.0 , 0.01)
for t in thresholds:
    actual_positive=(y_val==1)
    actual_negative=(y_val==0)
    
    predict_positive = (y_pred >=t)
    predict_negative = (y_pred < t)
    
    tp = (actual_positive & predict_positive).sum()
    tn = (predict_negative & actual_negative).sum()
    
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    r = tp / (tp + fn)
    p = tp / (tp + fp)
    
    f1 = 2 * (p*r)/(p+r) 
    
    scores.append((t, tp, tn, fp, fn,p,r,f1))


# In[84]:


columns = ['threshold', 'tp', 'tn', 'fp', 'fn', 'r', 'p','f1']
df_scores =pd.DataFrame(scores, columns=columns)


# In[85]:


df_scores


# In[86]:


from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer

# Assuming df_full_train is your training dataset with features and target

# Define the KFold object with 5 splits, shuffling the data, and setting a random seed
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Initialize a list to store AUC scores for each fold
auc_scores = []

# Initialize DictVectorizer
vectorizer = DictVectorizer()

# Iterate over different folds of df_full_train
for train_index, val_index in kf.split(df_full_train):
    # Split the data into train and validation sets based on the fold indices
    train_data, val_data = df_full_train.iloc[train_index], df_full_train.iloc[val_index]
    
    # Separate the features (X) and the target (y) variables for train and validation
    X_train, y_train = train_data.drop(columns=['reached_on_time_y_n']), train_data['reached_on_time_y_n']
    X_val, y_val = val_data.drop(columns=['reached_on_time_y_n']), val_data['reached_on_time_y_n']
    
    # Convert feature dataframes to dictionaries and then use DictVectorizer
    X_train_dict = X_train.to_dict(orient='records')
    X_val_dict = X_val.to_dict(orient='records')
    
    X_train_encoded = vectorizer.fit_transform(X_train_dict)
    X_val_encoded = vectorizer.transform(X_val_dict)
    
    # Initialize and train the Logistic Regression model
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X_train_encoded, y_train)
    
    # Predict probabilities on the validation set
    y_pred_proba = model.predict_proba(X_val_encoded)[:, 1]
    
    # Calculate the AUC score and append it to the list
    auc_score = roc_auc_score(y_val, y_pred_proba)
    auc_scores.append(auc_score)

# Calculate and print the mean AUC score across all folds
mean_auc = sum(auc_scores) / len(auc_scores)
print("Mean AUC:", mean_auc)


# In[87]:



# Define the C values to iterate over
C_values = [0.01, 0.1, 0.5, 10]

# Initialize KFold with the same parameters as previously
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Initialize lists to store mean and std scores
mean_scores = []
std_scores = []

# Iterate over different C values
for C in C_values:
    # Initialize a list to store AUC scores for each fold
    auc_scores = []
    
    # Iterate over different folds of df_full_train
    for train_index, val_index in kf.split(df_full_train):
        # Split the data into train and validation sets based on the fold indices
        train_data, val_data = df_full_train.iloc[train_index], df_full_train.iloc[val_index]

        # Separate the features (X) and the target (y) variables for train and validation
        X_train, y_train = train_data.drop(columns=['reached_on_time_y_n']), train_data['reached_on_time_y_n']
        X_val, y_val = val_data.drop(columns=['reached_on_time_y_n']), val_data['reached_on_time_y_n']

        # Convert feature dataframes to dictionaries and then use DictVectorizer
        X_train_dict = X_train.to_dict(orient='records')
        X_val_dict = X_val.to_dict(orient='records')

        X_train_encoded = vectorizer.fit_transform(X_train_dict)
        X_val_encoded = vectorizer.transform(X_val_dict)

        # Initialize and train the Logistic Regression model with the current C value
        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_train_encoded, y_train)

        # Predict probabilities on the validation set
        y_pred_proba = model.predict_proba(X_val_encoded)[:, 1]

        # Calculate the AUC score and append it to the list
        auc_score = roc_auc_score(y_val, y_pred_proba)
        auc_scores.append(auc_score)
    
    # Calculate the mean and standard deviation of AUC scores for the current C value
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    
    # Append the mean and std scores to the respective lists (rounded to 3 decimal digits)
    mean_scores.append(round(mean_auc, 3))
    std_scores.append(round(std_auc, 3))

# Print the results for each C value
for i, C in enumerate(C_values):
    print(f"C = {C}: Mean AUC = {mean_scores[i]}, Std = {std_scores[i]}")


# In[88]:
1.0.2 

import pickle


# In[91]:


output_file =f'model_C={C}.bin' 
output_file


# In[92]:


f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()


# In[ ]:




