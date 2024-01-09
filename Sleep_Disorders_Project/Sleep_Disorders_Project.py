#!/usr/bin/env python
# coding: utf-8

# In[193]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import svm

from sklearn.linear_model import LogisticRegressionCV
from sklearn import neural_network
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    recall_score,
    precision_score
)


from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as knn
import random
import warnings
from collections import defaultdict


# In[194]:


df = pd.read_csv("data\Sleep_health_and_lifestyle_dataset.csv")


# In[195]:


df.head()

df['Gender'] = df['Gender'].replace('Male', 0)
df['Gender'] = df['Gender'].replace('Female', 1)

df['BMI Category'] = df['BMI Category'].replace('Normal', 0)
df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 0)
df['BMI Category'] = df['BMI Category'].replace('Overweight', 1)
df['BMI Category'] = df['BMI Category'].replace('Obese', 2)


df['Sleep Disorder'] = df['Sleep Disorder'].fillna(0)
df['Sleep Disorder'] = df['Sleep Disorder'].replace('Sleep Apnea', 1)
df['Sleep Disorder'] = df['Sleep Disorder'].replace('Insomnia', 2)

df[['Systolic','Diastolic']] = df['Blood Pressure'].str.split("/",expand=True) .astype(float)

df = df.drop(['Person ID', 'Blood Pressure', 'Occupation'], axis=1)
# df = df.drop(['Blood Pressure', 'Occupation'], axis=1)


print(len(df))

# print(df['Sleep Disorder'].value_counts()[0])
# print(df['Sleep Disorder'].value_counts()[1])
# print(df['Sleep Disorder'].value_counts()[2])

print("0", 219/374)
print("1", 78/374)
print("2", 77/374)


# In[196]:


df_01 = df.copy()

df_01['Sleep Disorder'] = df_01['Sleep Disorder'].replace(2, 1)


# In[5]:


# #Correlation matrix

# fig = plt.figure(figsize=(10,10), dpi = 100)
# plt.title("Correlation Matrix of 'Sleep Health and Lifestyle'")
# sns.heatmap(df.corr(), annot = True, fmt = '.2f')


# In[197]:


X_01 = df_01.to_numpy()
#y = X_01[:, 10].reshape(374, 1)
y = X_01[:, 9]

#deleting y from training data
X_01 = np.delete(X_01, 9, 1) 


# In[198]:


# random.seed(11)
# train, test, split

X_train, X_test, y_train, y_test = train_test_split(
    X_01, y, test_size=0.2, shuffle = True, random_state = 6)


# In[199]:


def get_accuracy(pred, actual):
    d_actual = defaultdict(list)
    d_train= defaultdict(list)

    ovr_lab = [p for p, a in zip(pred, actual) if p == a]  

    cluster_actual, counts_actual = np.unique(actual, return_counts=True)
    actual_dict = dict(zip(cluster_actual, counts_actual))

    cluster_pred, counts_pred = np.unique(ovr_lab, return_counts=True)
    pred_dict = dict(zip(cluster_pred, counts_pred))

    one_missclass_rate = (pred_dict[0.0] / actual_dict[0.0])
    two_missclass_rate = (pred_dict[1.0] / actual_dict[1.0])
    three_missclass_rate = (pred_dict[2.0] / actual_dict[2.0])
    results = {"0": one_missclass_rate, "1": two_missclass_rate, "2": three_missclass_rate}
    return results

def get_accuracy_binary(pred, actual):
    d_actual = defaultdict(list)
    d_train= defaultdict(list)

    ovr_lab = [p for p, a in zip(pred, actual) if p == a]  

    cluster_actual, counts_actual = np.unique(actual, return_counts=True)
    actual_dict = dict(zip(cluster_actual, counts_actual))

    cluster_pred, counts_pred = np.unique(ovr_lab, return_counts=True)
    pred_dict = dict(zip(cluster_pred, counts_pred))

    one_missclass_rate = (pred_dict[0.0] / actual_dict[0.0])
    two_missclass_rate = (pred_dict[1.0] / actual_dict[1.0])
    results = {"0": one_missclass_rate, "1": two_missclass_rate}
    return results


def conf_matrix(y_pred, y_test, title):
    matrix = metrics.confusion_matrix(y_test, y_pred)
    matrix_display = ConfusionMatrixDisplay(confusion_matrix = matrix,
                              display_labels= [0, 1, 2])
    matrix_display.plot()
    plt.title(title)
    plt.savefig('plots/' + title)
    plt.show()
    
def conf_matrix_binary(y_pred, y_test, title):
    matrix = metrics.confusion_matrix(y_test, y_pred)
    matrix_display = ConfusionMatrixDisplay(confusion_matrix = matrix,
                              display_labels= [0, 1])
    matrix_display.plot()
    plt.title(title)
    plt.savefig('plots/' + title)
    plt.show()


# In[200]:



from sklearn.neighbors import KNeighborsClassifier as knn
### KNN FINAL MODEL ###

knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)

knn_accuracy = get_accuracy_binary(knn_y_pred, y_test)

print("KNN: \n")
print("No Disorder Accuracy: ", knn_accuracy['0'])
print("Sleep Apnea Accuracy: ", knn_accuracy['1'])

conf_matrix_binary(knn_y_pred, y_test, 'KNN Confusion Matrix (k=5)')


# In[201]:


## SINGLE SVM ###

svm_clf = svm.SVC(kernel='linear') # Linear Kernel
svm_clf.fit(X_train, y_train)
svm_y_pred = svm_clf.predict(X_test)

svm_accuracy = get_accuracy_binary(svm_y_pred, y_test)

print("SVM: \n")
print("No Disorder Accuracy: ", svm_accuracy['0'])
print("Sleep Apnea Accuracy: ", svm_accuracy['1'])

conf_matrix_binary(svm_y_pred, y_test, 'SVM Confusion Matrix')


# In[121]:


### SINGLE Gaussian Naive-Bayes ###

gnb = GaussianNB().fit(X_train, y_train) 
y_pred = gnb.predict(X_test) 

gnb_accuracy = get_accuracy_binary(y_pred, y_test)

print("Gaussian Naive-Bayes: \n")
print("No Disorder Accuracy: ", gnb_accuracy['0'])
print("Sleep Apnea Accuracy: ", gnb_accuracy['1'])

conf_matrix_binary(y_pred, y_test, 'Gaussian Naive-Bayes Confusion Matrix')


# In[128]:


# from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)


### LOGISTIC REGRESSION CROSS VALIDATION ###

clf = LogisticRegressionCV(cv=5, random_state=0).fit(X_train, y_train)
lg_y_pred = clf.predict(X_test)

lg_accuracy = get_accuracy_binary(lg_y_pred, y_test)

print("Logistic Regression: \n")
print("No Disorder Accuracy: ", lg_accuracy['0'])
print("Sleep Apnea Acuracy: ", lg_accuracy['1'])

conf_matrix_binary(lg_y_pred, y_test, 'Logistic Regression Confusion Matrix')


# In[130]:


### Decision Tree ###

dtree_model = DecisionTreeClassifier().fit(X_train, y_train) 
dtree_pred = dtree_model.predict(X_test) 

dtree_accuracy = get_accuracy_binary(dtree_pred, y_test)

print("Decision Tree: \n")
print("No Disorder Accuracy: ", dtree_accuracy['0'])
print("Sleep Apnea Accuracy: ", dtree_accuracy['1'])

conf_matrix_binary(dtree_pred, y_test, 'Decision Tree Confusion Matrix')


# In[131]:


from sklearn import tree

dtree_model_binary = DecisionTreeClassifier(max_leaf_nodes = 10).fit(X_train, y_train) 
y_pred = dtree_model_binary.predict(X_test)

print(get_accuracy_binary(y_pred, y_test))

fig_full = plt.figure(figsize=(12,10))
cart_plot_full = tree.plot_tree(dtree_model_2, 
                   feature_names=list((df_01.columns)),  
                   class_names= ['0','1', '2'],
                   filled=True)
plt.title("CART Model")
fig_full.savefig("plots/binary_CART_10.png")


# In[132]:


### RANDOM FOREST FINAL MODEL ###

rf_final = RandomForestClassifier(n_estimators = 25, random_sta
                                  te = 4).fit(X_train, y_train)
rf_pred = rf_final.predict(X_test)

rf_accuracy = get_accuracy_binary(rf_pred, y_test)

print("Random Forest: \n")
print("No Disorder Accuracy: ", rf_accuracy['0'])
print("Sleep Apnea Acuracy: ", rf_accuracy['1'])

conf_matrix_binary(rf_pred, y_test, 'Random Forest Confusion Matrix')


# In[ ]:





# In[ ]:





# In[13]:


#### MULTI-CLASSIFICATION ###


# In[203]:


X = df.to_numpy()
y = X[:, 9]
X = np.delete(X, 9, 1) 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle = True, stratify = y)


# In[32]:


# # One Vs Rest SVM

# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC

# clf = OneVsRestClassifier(SVC()).fit(X_train, y_train)
# clf_y_pred = clf.predict(X_test)

# print("Accuracy:", accuracy)
# print("F1 Score:", f1)


# In[31]:


# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# cm = confusion_matrix(y_test, clf_y_pred, labels = clf.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                                display_labels=clf.classes_)
# disp.plot()
# plt.title('Confusion matrix for OvR classifier')
# plt.show()


# print("No Disorder Missclassification: ", one_missclass_rate)
# print("Sleep Apnea Missclassification: ", two_missclass_rate)
# print("Insomnia Missclassification: ", three_missclass_rate)


# In[34]:


### KNN Hyperparameter Tuning ###

# from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neighbors import KNeighborsClassifier

k_values = [i for i in range (1,31)]
scores = []


for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=5)
    scores.append(np.mean(score))
    
    
sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.grid(zorder=0)
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.title("Mean CV Accuracy Relative to K Value")
plt.savefig('plots/knn_vs_k')


# In[35]:


### KNN FINAL MODEL ###

knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)

knn_accuracy = get_accuracy(knn_y_pred, y_test)

print("KNN: \n")
print("No Disorder Accuracy: ", knn_accuracy['0'])
print("Sleep Apnea Accuracy: ", knn_accuracy['1'])
print("Insomnia Accuracy: ", knn_accuracy['2'])

conf_matrix(knn_y_pred, y_test, 'KNN Confusion Matrix (k=5)')


# In[ ]:


# from sklearn.model_selection import GridSearchCV 
  
# # defining parameter range 
# param_grid = {'C': [0.1, 1, 10, 100, 1000],  
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
#               'kernel': ['rbf']}  
  
# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# # fitting the model for grid search 
# grid.fit(X_train, y_train) 



# # print best parameter after tuning 
# # print(grid.best_params_) 
  
# # print how our model looks after hyper-parameter tuning 
# # print(grid.best_estimator_) 

# # grid_predictions = grid.predict(X_test) 
# print(grid_predictions)
# print(get_accuracy(grid_predictions, X_test))


# In[ ]:


# c_values = [1, 10, 100, 1000]
# accuracy = []


# for c in c_values:
#     svm_clf = svm.SVC(kernel='linear', C = c).fit(X_train, y_train)
#     score = cross_val_score(svm_clf, X_train, y_train, cv=5)
#     scores.append(np.mean(score))
    
    
# sns.lineplot(x = c_values, y = scores, marker = 'o')
# plt.grid(zorder=0)
# plt.xlabel("Number of Neighbors")
# plt.ylabel("Accuracy")
# plt.title("Mean CV Accuracy Relative to K Value")
# plt.savefig('plots/svm_vs_c')


# In[36]:


## SVM ###

svm_clf = svm.SVC(kernel='linear') # Linear Kernel
svm_clf.fit(X_train, y_train)
svm_y_pred = svm_clf.predict(X_test)

svm_accuracy = get_accuracy(svm_y_pred, y_test)

print("SVM: \n")
print("No Disorder Accuracy: ", svm_accuracy['0'])
print("Sleep Apnea Accuracy: ", svm_accuracy['1'])
print("Insomnia Accuracy: ", svm_accuracy['2'])

conf_matrix(svm_y_pred, y_test, 'SVM Confusion Matrix')
    


# In[37]:


### Gaussian Naive-Bayes ###

gnb = GaussianNB().fit(X_train, y_train) 
y_pred = gnb.predict(X_test) 

gnb_accuracy = get_accuracy(y_pred, y_test)

print("Gaussian Naive-Bayes: \n")
print("No Disorder Accuracy: ", gnb_accuracy['0'])
print("Sleep Apnea Accuracy: ", gnb_accuracy['1'])
print("Insomnia Accuracy: ", gnb_accuracy['2'])

conf_matrix(y_pred, y_test, 'Gaussian Naive-Bayes Confusion Matrix')


# In[41]:


### Decision Tree ###

# dtree_model = DecisionTreeClassifier().fit(X_train, y_train) 
# dtree_pred = dtree_model.predict(X_test) 

# dtree_accuracy = get_accuracy(dtree_pred, y_test)

print("Decision Tree: \n")
print("No Disorder Accuracy: ", dtree_accuracy['0'])
print("Sleep Apnea Accuracy: ", dtree_accuracy['1'])
print("Insomnia Accuracy: ", dtree_accuracy['2'])

# conf_matrix(dtree_pred, y_test, 'Decision Tree Confusion Matrix')


# In[39]:


from sklearn import tree

dtree_model_2 = DecisionTreeClassifier(max_leaf_nodes = 10).fit(X_train, y_train) 
y_pred = dtree_model_2.predict(X_test)

print(get_accuracy(y_pred, y_test))

fig_full = plt.figure(figsize=(12,10))
cart_plot_full = tree.plot_tree(dtree_model_2, 
                   feature_names=list((df_01.columns)),  
                   class_names= ['0','1', '2'],
                   filled=True)
plt.title("CART Model")
fig_full.savefig("plots/CART_10.png")


# In[204]:


### DECISION TREE Hyperparameters ###
dt_accuracy = []

max_depth = list(range(2, 20))

for n in num_leafs:
    dtree_model_2 = DecisionTreeClassifier(max_depth = n).fit(X_train, y_train) 
    y_pred = dtree_model_2.predict(X_test)
    error = metrics.accuracy_score(y_test, y_pred)
    dt_accuracy.append(error)


# In[205]:


plt.figure(figsize = (6, 6))

plt.plot(max_depth, dt_accuracy)
plt.xlim(0, 7)
plt.title('')
plt.xlabel('Number of Leaf Nodes')
plt.ylabel('Accuracy')
# plt.savefig("random_forest_num_trees.png")
plt.show()


# In[206]:


### Random Forest Hyperparameters ###
rf_accuracy = []

num_trees = list(range(1, 100, 5))

for n in num_trees:
    rf = RandomForestClassifier(n_estimators = n, random_state = 4).fit(X_train, y_train)
    pred = rf.predict(X_test)
    error = metrics.accuracy_score(y_test, pred)
    rf_accuracy.append(error)


# In[207]:


plt.figure(figsize = (6, 6))

plt.plot(num_trees, rf_accuracy)
plt.title('Random Forest Accuracy Relative to Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.savefig("random_forest_num_trees.png")
plt.show()


# In[46]:


### RANDOM FOREST FINAL MODEL ###

rf_final = RandomForestClassifier(n_estimators = 25, random_state = 4).fit(X_train, y_train)
rf_pred = rf_final.predict(X_test)

rf_accuracy = get_accuracy(rf_pred, y_test)

print("Random Forest: \n")
print("No Disorder Accuracy: ", rf_accuracy['0'])
print("Sleep Apnea Acuracy: ", rf_accuracy['1'])
print("Insomnia Accuracy: ", rf_accuracy['2'])

conf_matrix(rf_pred, y_test, 'Random Forest Confusion Matrix')


# In[47]:


### LOGISTIC REGRESSION CROSS VALIDATION ###

clf = LogisticRegressionCV(cv=5, random_state=0).fit(X_train, y_train)
lg_y_pred = clf.predict(X_test)

lg_accuracy = get_accuracy(lg_y_pred, y_test)

print("Logistic Regression: \n")
print("No Disorder Accuracy: ", lg_accuracy['0'])
print("Sleep Apnea Acuracy: ", lg_accuracy['1'])
print("Insomnia Accuracy: ", lg_accuracy['2'])

conf_matrix(lg_y_pred, y_test, 'Logistic Regression Confusion Matrix')


# In[ ]:


#SVM hyperparamater tuning with C


# In[80]:


#knn, svm, gnb, dt, rf

# print(rf_accuracy)
# print(lg_accuracy)
# print(knn_accuracy)
# print(svm_accuracy)
# print(dtree_accuracy)

models =('random_forest', 'logistic_regression', 'KNN', 'SVM', 'decision_tree')
    
results = {'no_disorder': [0.9848, 0.9848, 0.9545, 0.9848, 0.9848],
          'sleep_apnea':[0.8333, 0.8333, 0.7917, 0.9167, 0.8333],
          'insomnia':[0.8261,0.7826, 0.7826, 0.7826, 0.6957]}

results_df = pd.DataFrame(results)

print(results)



# In[161]:


# models =('random_forest', 'logistic_regression', 'KNN', 'SVM', 'decision_tree')
    
# results = {'no_disorder': [0.9848, 0.9848, 0.9545, 0.9848, 0.9848],
#           'sleep_apnea':[0.8333, 0.8333, 0.7917, 0.9167, 0.8333],
#           'insomnia':[0.8261,0.7826, 0.7826, 0.7826, 0.6957]}


results = {'random_forest':[0.9848, 0.8333, 0.8261],
          'logistic_regression': [0.9848, 0.8333, 0.7826],
          'KNN': [0.9545, 0.7917, 0.7826],
          'SVM': [0.9848, 0.9167, 0.7826],
          'decision_tree': [0.9848, 0.8333, 0.6957],
          'naive_bayes': [0.9697, 0.8333, 0.7826]}

categories = ['no_disorder', 'sleep_apnea', 'insomnia']

width = 0.1
x = np.arange(3) 
plt.ylim(0.6, 1)
plt.bar(x-0.2, results['random_forest'], width, color='firebrick') 
plt.bar(x-0.1, results['logistic_regression'], width , color='orange') 
plt.bar(x, results['KNN'], width, color='gold') 
plt.bar(x+0.1, results['SVM'], width, color='grey') 
plt.bar(x+0.2, results['decision_tree'], width, color='bisque') 
plt.bar(x+0.3, results['naive_bayes'], width, color='darkkhaki') 




plt.xticks(x, ['No Disorder', 'Sleep Apnea', 'Insomnia']) 
plt.title("Multi-Classification Model Accuracy per Dependent Category") 
plt.ylabel("Accuracy") 
plt.legend(['Random Forest', 'Logistic Regression', 'KNN', 'SVM', 'Decision Tree', 'Naive-Bayes'], bbox_to_anchor=(1.04, 1), loc="upper left") 
plt.savefig('plots/final_results', bbox_inches='tight')


# In[160]:


### BINARY ###

#knn, svm, gnb, dt, rf

# print(rf_accuracy)
# print(lg_accuracy)
# print(knn_accuracy)
# print(svm_accuracy)
# print(dtree_accuracy)
print(gnb_accuracy)

# models =('random_forest', 'logistic_regression', 'KNN', 'SVM', 'decision_tree', 'Naive-Bayes')
    
# # results_binary = {'no_disorder': [0.9565, 0.9565, 0.9130, 0.9565, 0.9565],
# #           'disorder':[0.8621, 0.9310, 0.8276, 0.9310, 0.8621]}


results_binary = {'random_forest':[0.9565, 0.8621],
          'logistic_regression': [0.9565, 0.9310],
          'KNN': [0.9130, 0.8276],
          'SVM': [0.9565,0.9310],
          'decision_tree': [0.9565, 0.8621],
                  'naive_bayes':[0.9565, 0.9310]}


categories = ['no_disorder', 'disorder']

width = 0.1
x = np.arange(2) 
plt.ylim(0.6, 1)
plt.bar(x-0.2, results_binary['logistic_regression'], width , color='orange') 
plt.bar(x-0.1, results_binary['SVM'], width, color='grey') 
plt.bar(x, results_binary['decision_tree'], width, color='bisque') 
plt.bar(x+0.1, results_binary['KNN'], width, color='gold') 
plt.bar(x+0.2, results_binary['random_forest'], width, color='firebrick') 
plt.bar(x+0.3, results_binary['naive_bayes'], width, color='darkkhaki') 






plt.xticks(x, ['No Disorder', 'Disorder']) 
plt.title("Classification Model Accuracy per Dependent Category") 
plt.ylabel("Accuracy") 
plt.legend(['Logistic Regression','SVM', 'Decision Tree', 'KNN', 'Random Forest', 'Gaussian Naive-Bayes'], bbox_to_anchor=(1.04, 1), loc="upper left") 
plt.savefig('plots/final_results_binary', bbox_inches='tight')


# In[ ]:




