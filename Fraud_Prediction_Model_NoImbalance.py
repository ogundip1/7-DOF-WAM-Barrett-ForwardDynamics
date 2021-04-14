#!/usr/bin/env python
# coding: utf-8

# # Preamble: Project Task and Requirements
#  
# <b>Statement of Problem:</b> *Identifying Fraudulent Activities*
# 
# Company XYZ is an e-commerce site that sells fashion apparel. 
# The task is to build a machine learning model that predicts whether a user has a high probability of using the site to perform some fraudulent activity or not. For example, using stolen credit cards, doing money laundry, cloning access, etc. We only have information about the user’s first transaction on the site and based on that, we need to make a classification if it is fraud or no fraud. The provided dataset is “fraud_data.csv” and “IpAddress_to_Country.csv” is available for use, both as imported within.
# 
# The aim is to help their risk team with a tool to avoid or predict frauds via machine learning – predicting whether the first transaction of a new user is fraudulent or not.
# 
# These are the tasks required by the e-commerce company:\
# (1)	For each user, determine her country based on the numeric IP address.
# 
# (2)	Build a model to predict whether an activity is fraudulent or not. Assist with explanation how different assumptions about the cost of false positives vs false negatives would impact the model.
# 
# (3)	Our boss is a bit worried about using a model she does not understand for something as important as fraud detection. Could you come up with explanation for her on how the model is making the predictions? Not from a mathematical perspective (she could not care less about that), but from a user perspective. What kinds of users are more likely to be classified as at risk? What are their characteristics?
# 
# (4)	Let us say you now have this model which can be used live to predict in real time if an activity is fraudulent or not. From a product perspective, how would you recommend us to use it? That is, what kind of different user experiences would you build based on the model output.
# 

# In[1]:


import pandas as pd
import numpy as np
import sklearn
import scipy
import time
import matplotlib.pyplot as plt
import seaborn as sn
import random
from random import randrange
from random import seed 
from math import exp
from pandas import Series, DataFrame
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud", "Misclassified Fraud"]


# # Datasets Import and Look-up

# In[2]:


#pd.set_option('display.max_rows', 200)
df = pd.read_csv('fraud_data.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df2 =  pd.read_csv('IpAddress_to_Country.csv')
df2


# # Code to determine user's country based on the numeric IP address 

# In[5]:


tic = time.time()
IP_country = df2.country.to_numpy(dtype='object')
IP = np.array(df2[['lower_bound_ip_address','upper_bound_ip_address','country','lower_bound_ip_address']])
IP2 = np.zeros(shape=(len(IP),3))
IPunique = np.unique(IP_country)
#print(FraudIP.to_numpy(dtype ='float64')) 
Fraud = df.ip_address.to_numpy(dtype ='float64')
Fraud2 = np.zeros(shape=(len(Fraud),2))

for j in range(len(IP)):
    for k in range(len(IPunique)):
        if IP_country[j] == IPunique[k]:
            IP2[j,2] = k
            IP2[j,0] = IP[j,0]
            IP2[j,1] = IP[j,1] 
toc = time.time()
print(toc-tic, 'sec Elapsed')
#return IP2

    


# # Code matching Country IP integers with fraud_data.csv file
# *Results saved out and stored as csv file, to be used as the updated/prepared dataset. Takes too long to run*

# In[6]:


#    tic = time.time()
#    for i in range(len(Fraud)):
#        for j in range(len(IP)):
#            if IP2[j,0] <= Fraud[i]: 
#                if Fraud[i] <= IP2[j,1]:
#        #if IP2[j,0] <= Fraud[i]:
#            #if IP2[j,1] >= Fraud[i,0]:
#                    Fraud2[i,1] = IP2[j,2]
#                    Fraud2[i,0] = Fraud[i]
#    toc = time.time()
#    print(toc-tic, 'sec Elapsed')
#    return Fraud2 

### Saving the Fraud2 array as csv file representing Country IP and Country Integer
#    np.savetxt("CountryIP2Integers.csv", Fraud2, delimiter=",")
### Converting array of Country unique name/array number to dataframe and to csv file
#    dff = pd.DataFrame(IPunique)
#    dff.to_csv('CountryInt_to_Name.csv', index=False)


# # Reading Prepared/Updated Datasets

# In[7]:


# Reading and uploading the updated/prepared/renamed CountryIP2Integers.csv file
df3 =  pd.read_csv('CountryIP2Integers_2.csv')
df3


# In[8]:


# Reading and uploading the updated/prepared/renamed CountryInt_to_Name.csv file
df4 = pd.read_csv('CountryInt_to_Name2.csv')
df4


# # Exploratory Data Analysis (EDA)

# In[9]:


schema_df = pd.read_csv('fraud_data.csv')
schema_df


# In[10]:


## Using purchase_time and signup_time to calculate "signup_duration" --> dt.total_seconds()

# convert the 'purchase_time and signup_time' columns to datetime format, instead of objects (strings) 
df['purchase_time']= pd.to_datetime(df['purchase_time']) 
df['signup_time']= pd.to_datetime(df['signup_time']) 

signup_duration = (df.purchase_time - df.signup_time).dt.total_seconds()


# In[11]:


# Using df.insert() to add columns for Signup Duration and CountryIP Integers
df.insert(3, "signup_duration", signup_duration[0:(len(signup_duration))], True) 
df.insert(11, "CountryIP", df3.Country_IP[0:(len(signup_duration))], True) 
df.insert(12, "Country_Int", df3.Country_Integer[0:(len(signup_duration))], True) 
df


# In[12]:


df.info()


# In[13]:


## Checking if any null value in dataset
df.isnull().values.any()


# In[14]:


# Removing rows where/with CountryIP having Zeros
df = df[df.CountryIP != 0]
df.shape


# In[15]:


## Transaction Class Distribution on Bar plot

count_classes = pd.value_counts(df['class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[16]:


### Checking/experimenting on "signup_duration" which are less than 300 seconds... 
## Trying to see if there is a quick/evident connection between shorter signup_duration and fraudulent cases

df.loc[df['signup_duration'] <= 300, 'less_300s'] = 1 
df.loc[df['signup_duration'] > 300, 'less_300s'] = 0
#pd.set_option('display.max_rows', 200)
df.shape
df


# In[17]:


## Get the Fraud and the normal dataset 

fraud = df[df['class']==1]

normal = df[df['class']==0]

print(fraud.shape,normal.shape)


# In[18]:


## Get the Fraud and the normal dataset based on assumed 300s signup_duration for fraud 
## Simply for checking only, just to see the assumed correlation..

fraud2 = df[df['less_300s']==1]

normal2 = df[df['less_300s']==0]

print(fraud2.shape,normal2.shape)


# In[19]:


## Analyzing more on amount of information from the transaction data
#How much are the amount of money reported in different transaction classes..
print('Total Fraud Transaction ($) = ', fraud.purchase_value.sum())
fraud.purchase_value.describe()


# In[20]:


print('Total Normal Transaction ($) = ', normal.purchase_value.sum())
normal.purchase_value.describe()


# In[21]:


### Transaction Amount by Class ###  
# Visual representation..

amount = [normal.purchase_value.sum(), fraud.purchase_value.sum()]

plt.title("Transaction Amount by Class")
plt.xlabel("Class")
plt.ylabel("Transaction Amount ($)")
plt.bar(range(len(amount)), amount, color='royalblue', alpha=1, width=0.5)
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(range(2), LABELS)
plt.show()


# In[22]:


### Fraud and Normal Purchase Amounts by Class
# Visual representation..

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per Purchase by Class')
bins = 50
ax1.hist(fraud.purchase_value, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.purchase_value, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 200))
plt.yscale('log')
plt.show();


# In[23]:


# Checking if fraudulent transactions occur more often during certain signup duration..
# Find out with a visual representation..

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Signup Duration vs Amount by class')
ax1.scatter(fraud.signup_duration, fraud.purchase_value)
#ax1.scatter(fraud.signup_duration, fraud.class) # or fraud.less_300s
ax1.set_title('Fraud')
ax2.scatter(normal.signup_duration, normal.purchase_value)
#ax2.scatter(normal.signup_duration, normal.class) # or fraud.less_300s
ax2.set_title('Normal')
plt.xlabel('Signup Duration (in Seconds)')
plt.ylabel('Amount')
plt.show()


# In[24]:


#Determine the % of fraud to valid transactions in the dataset (original)

Fraud = df[df['class']==1]

Valid = df[df['class']==0]

outlier_fraction = len(Fraud)/float(len(Valid)) ## Percentage of Fraud

print('Percentage of Fraud = ', outlier_fraction)
print("Fraud Cases : {}".format(len(Fraud)))
print("Valid Cases : {}".format(len(Valid)))


# # Dataset Clean-Up (Additional Data Preparation)
# *Converting strings to integers and dropping columns of dataset that are now considered irrelevant*

# In[25]:


## Taking/Using just some % sample of the data [Avoid long processing and show Predictor (Model) Accuracy]

data1= df.sample(frac = (5/100),random_state=1) ### here, (10/100) = 10% Case study, 1500 customers
# 0.1% == case study on 150 random customers
data1.shape


# In[26]:


#### Dataset Clean-Up #####

## Conversion of strings in data to integer representation for data processing
data1.sex[data1.sex == 'M'] = 1
data1.sex[data1.sex == 'F'] = 0
data1.source[data1.source == 'SEO'] = 0
data1.source[data1.source == 'Direct'] = 1
data1.source[data1.source == 'Ads'] = 2
data1 


# In[27]:


#### Dataset Clean-Up #####

## Converting the data in object to numerical
data1['source'] = pd.to_numeric(data1['source'])
data1['sex'] = pd.to_numeric(data1['sex'])


# In[28]:


#### Dataset Clean-Up #####

## Dropping columns of dataset that are now considered irrelevant to this analysis:
data1 = data1.drop(columns=['user_id','signup_time','purchase_time','CountryIP','ip_address','device_id','browser','less_300s'])
data1


# In[29]:


data1.info()


# # Analysis with above Fraction of Data and Correlation Check
# *To Determine Impact/Validity of Predictor on Business Outcome*

# In[30]:


#Determine the number of fraud and valid transactions in the REDUCED dataset
#Check the percentage of the fraud cases if it is still about the same as original large data

Fraud = data1[data1['class']==1]

Valid = data1[data1['class']==0]

outlier_fraction = len(Fraud)/float(len(Valid))

print('Percentage of Fraud = ', outlier_fraction)
print("Fraud Cases : {}".format(len(Fraud)))
print("Valid Cases : {}".format(len(Valid)))


# In[31]:


#### Correlation Check ####

#Get correlations of each features in dataset
corrmat = data1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
plt.title("Heatmap of Correlation")
#plot heat map
g=sn.heatmap(data1[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# # Feature Engineering

# In[32]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing 

dataset = data1

# Normalizing dataset 
x = dataset.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dataset = pd.DataFrame(x_scaled, columns=dataset.columns)

#Create Independent and Dependent Features
columns = dataset.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["class"]]
# Store the variable we are predicting 
target = "class"
# Define a random state 
state = np.random.RandomState(42)
X = dataset[columns]
Y = dataset[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Print the shapes of X & Y
print('Shape - Independent = ', X.shape)
print('Shape - Dependent/Target = ', Y.shape)


# # Prediction and Model Performance/Analysis
# 

# In[33]:


from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression 
#from sklearn.model_selection import KFold 

# Using Sklearn Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

#Checking model accuracy:
accuracy = sklearn.metrics.accuracy_score(Y_test, Y_pred)
print('Accuracy = ', accuracy)

### IGNORE: Cross-Validation makes no significant difference ###
#xx = X
#yy = Y
#kf = KFold(n_splits=2)
#kf.get_n_splits(xx)
#2

#print(kf)
#KFold(n_splits=2, random_state=None, shuffle=False)

#for train_index, test_index in kf.split(xx):
#    print("TRAIN:", train_index, "TEST:", test_index)
#    [X_train, X_test] = xx[train_index], xx[test_index]
#    [y_train, y_test] = yy[train_index], yy[test_index]


# In[34]:


#### Model Analysis and Interpretations #### 

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)
conf_matrix = pd.DataFrame(data=cm, columns =['Predicted:0', 'Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
plt.title("Confusion Matrix")
sn.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[35]:


#### Model Analysis and Interpretations ####

TN = cm[0,0] # True Negative
TP = cm[1,1] # True Positive
FN = cm[1,0] # False Negative
FP = cm[0,1] # False Positive
Sensitivity = TP/float(TP+FN)
Specificity = TN/float(TN+FP)

print('Recall = ', Sensitivity, '\n',
      'Specificity = ', Specificity)

#Model Evaluation Statistics
print('Accuracy of the Model = TP+TN/(TP+TN+FP+FN) = ', (TP+TN)/float(TP+TN+FP+FN), '\n',
     'Misclassification = 1 - Accuracy = ', 1 - ((TP+TN)/float(TP+TN+FP+FN)), '\n', 
     'Sensitivity (True Positive Rate) = TP/float(TP+FN) = ', TP/float(TP+FN), '\n',
     'Specificity (True Negative Rate) = TN/float(TN+FP) = ', TN/float(TN+FP), '\n',
     'Precision/Positive Predictive Value = TP/(TP+FP) = ', TP/float(TP+FP), '\n', 
     'Negative Predictive Value = TN/(TN+FN) = ', TN/float(TN+FN), '\n', 
     'Positive Likelihood Ratio = Sensitivity/(1-Precision) = ', float(Sensitivity)/(1-Specificity), '\n', 
     'Negative Likelihood Ratio = (1-Sensitivity)/Precision = ', (1-Sensitivity)/Specificity)

# Precision --> Specificity and Recall--> Sensitivity 

# Recall in the context is also referred to as the true positive rate or sensitivity
# True negative rate is also called specificity. 
# Precision is also referred to as positive predictive value (PPV).


# In[36]:


#### Result Analysis and Impact of Predictor Accuracy on Business Revenue ####

dd = [normal.purchase_value.sum(), fraud.purchase_value.sum(), fraud.purchase_value.sum()*(1 - ((TP+TN)/float(TP+TN+FP+FN)))]

plt.title("Comparison of Potential Transaction Loss by Misclassification")
#major_ticks = np.arange(0, max(dd), 500000)
#yticks(major_ticks)
plt.xlabel("Class")
plt.ylabel("Transaction Amount ($)")
plt.bar(range(len(dd)), dd, color='royalblue', alpha=1, width=0.5)
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(range(3), LABELS)
plt.show()


# # Impacts and Summary of Results
# From a business perspective, it is important to point out that: 
# 
# (1) Fraudulent transactions constitute about 10% of the e-commerce company's transaction, which is quite a bit (see graph above). 
# 
# (2) The business revenue associated to this is about USD 450,243; it would be important to have a near accurate model/predictor to mitigate this. However, the business would not want to lose this much of revenue if a predictor would be adjudging (wrongly) a normal transaction as fraudulent; that is quite a bit of revenue to toy with.
# 
# (3) Taking this into consideration, we present here a model with an accuracy of 91.6% under 5% of the dataset, representing a customer ratio of 6400. With this, the business is only at the risk of 8.4% misclassification of fraudulent transactions, accounting for about USD 37,820 of its approximate USD 4,735,028 revenue, with respect to this provided data. This represents 0.8% of the business revenue, a significant trade-off for business integrity and customers' protection that would be provided by the model/predictor being proposed.
# Furthermore, the model still performs pretty well with 1% of the dataset (a case study with 1282 transactions only), showing an accuracy of 80%. This represents a risk of 20% misclassification of fraudulent transactions, accounting for about USD 90,049 of the USD 4,735,028 revenue, with respect to this provided data. This represents about 2% of the business revenue, which is still a good trade-off for business integrity and customers' protection.
# 
# (4) As seen from the correlation map, the data exploration also provided us with information on the kind of users that are more likely to be classified as at risk, and what are their characteristics. Mainly those with shorter signup duration and likely those who are male with a combination of higher purchase value and "source" of user (See the Correlation Map to explain). 
# 
# (5) Lastly, the model can be deployed in real time to predict if a (first-time) user's activity is fraudulent or not. From a product perspective, how would you use it? That is, what kind of different user experiences would you build based on the model output? \
# *Answer: I would propose a mobile app, which is able to track users more accurately. If the user is logging in for the first time, we can deploy a two-step verification procedure. Based on the data analysis results, we know that most of the fraudulent users spend less time (sign-up duration), we could build in a timer as part of the application to check duration before transaction, and flagging up appropriately. Also, an embedded application for detecting "source" of first time user. We can create a rule-based first layer; saying for example, if it is direct source, check then the sign-up duration, and maybe I.P address or gender (IF Source this...then check Duration...then check I.P/Gender). If these three queries fall through on positive, then we could send the user through the fraud-predictor model as a second layer verification. However, if the first layer's three queries miss some set query threshold, no need to send the user's transaction to the fraud-predictor layer. You want user-experience to be pleasant with minimal backend processsing or checks. Equally, you want the "user-experience application" to flag potential fraudulent transactions and send them to the risk team for investigation. This two-layer verification scheme will provide balanced user experience...based on the model output as discovered fron the data analysis.*  
# 
# *Potential impact: the model will reduce fraud by approximately 90%*
# 
# ####  This is the end of Main Result  ######
# 
# For Presentation: 2-3-4 slides of summary available for Business Executive presentation. 
# Mobile app able to track individual more accurately, than browser or laptop. 
# 

# # Experimenting with Isolation Detection Methods:
# Checking perhaps Isolation/Outlier algorithm (classifier) produces closer result. I would not expect an outlier detection algorithm to perform closer or any better than Logistic Regression for this particular data case we have. The ratio of fraud cases in the dataset is significant at over 10%. For instance, if the ratio of fraud cases was something in the range of 1-3 percent over a large dataset like this, I would have expected an outlier detection predictor to perhaps behave better, or as good as logistic regression. Additionally, outlier predictor would find this dataset not so ideal, because there is no single (or few) independent variable evident enough to be associated to the fraudulent transactions. Signup_duration would have been a good one, but it is not that significant/dominant enough.  

# In[37]:


##Define the outlier detection methods

classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction,random_state=state, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=5, algorithm='auto', 
                                              leaf_size=10, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
  #  "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
   #                                      max_iter=-1, random_state=)
   
}

n_outliers = len(Fraud)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    #Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:    
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
    #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != Y).sum()
    # Run Classification Metrics
    print("{}: {}".format(clf_name,n_errors))
    print("Accuracy Score :")
    print(accuracy_score(Y,y_pred))
    print("Classification Report :")
    print(classification_report(Y,y_pred))


# # Written Logistic Regression Implementation
# This is quite academic, simply an academic exercise. You could just ignore. Below is a regression code I worked on before; I have modified it a bit to test this problem with it. Its performance could be compared to the in-built sklearn model. We could use this for academic purpose when you start the Glomacs Training; for Track 2, for those really interested in the mathematics of how regression algorithm actually works. I have some materials I could use to teach this, with only a fair amount of math, so it is not required for a student to be so deep in math. Although some basics in math will be required; at least to first understand Linear Regression in the sense of y = mx + b, before moving to why logistic regression, and then I could take such interested students up to Classification methods - multivariate.

# In[38]:


# Calc. the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
        return minmax


# In[39]:


# Normalizing - rescaling dataset columns to range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0])/(minmax[i][0])


# In[40]:


# Splittig dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)/n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            #index = randrange(len(dataset_copy))
            index = randrange(1, len(dataset_copy), 1)
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold) 
    return dataset_split


# In[41]:


# Calculating accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
            return correct / float(len(actual))*100.0


# In[42]:


# Evaluating an algorithm using cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args): # Remove dataset and n_folds from function here, replace straight with train set, test set.
    folds = cross_validation_split(dataset, n_folds) ### Don't call this function
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy[-1] = None
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores        


# In[43]:


# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0] #here means, weight(i=O) = coefficients[0], since weight = 0, h(w,x) = 0, so initial yhat(i=0) = 0.
    for i in range(len(row-1)):
        yhat += coefficients[i+1]*row[i] #now computing the subsequent yhats, starting from the initialization, and adding them up (yhat0+yhat1+yhat2+yhat3+...) 
    #E.g yhat(i=0 +1) = coefficients[1]*row[0]
    #    yhat(i=1 +1) = coefficients[2]*row[1]
    #    yhat(0) = 0 --- Initialized
    return 1.0/(1 + exp(-yhat))


# In[44]:


# Estimate logistic regression coefficients using stochastic gradient descent

def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(dataset[0]))] #initializing first coef to zero 
    # And here below we are calculating the coef for each number of vectors that we have (each row's coef)
    for epoch in range(n_epoch): 
        for row in train:
            yhat = predict(row, coef)
            error = row[-1] - yhat # row[-1] - h(w,x)[predicted] --> [yi - h(w,x)]
            coef[0] = coef[0] + l_rate*error*yhat*(1.0-yhat) #gradient descent formular for calc. the coefs/weights..
            for i in range(len(row)-1):
                coef[i+1] = coef[i+1] + l_rate*error*yhat*(1.0-yhat)*row[i] #calc. for each to return the optimum coef value..
    return coef


# In[45]:


# Logistic Regression Algorithm with Stochastic Grad. Descent
def logistic_regression(train, test, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row,coef)
        yhat = round(yhat)
        predictions.append(yhat)
    return(predictions)


# In[46]:


# Test the Logistic regression algorithm on the dataset
n_folds = 2
l_rate =  0.1
n_epoch = 5

scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
print('scores: %s' % Scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

#predictions = logistic_regression(dataset, dataset, l_rate, n_epoch)
#print('predictions: %s' % predictions)
#print('Mean Accuracy: %.3f%%' % (sum(predictions)/float(len(predictions))))


# In[ ]:




