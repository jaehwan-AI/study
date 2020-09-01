# import library
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

import tensorflow as tf
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# seed
import os
seed = 123
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# load dataset
diabetes = load_diabetes()

# dataframe
data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
target = pd.DataFrame(diabetes.target, columns=["Target"])
df = pd.concat([data, target], axis=1)
df

df.describe()

# data distribution
plt.figure(figsize=(30,30))
for i in range(len(df.columns)):
    plt.rc('font', size=25)
    plt.subplot(5,3,i+1)
    df[df.columns[i]].plot.hist()
    plt.title(f'{df.columns[i]}')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

# dataset
x = diabetes.data
y = diabetes.target

# Decision Tree
def tree_model(data, label):
    tree = DecisionTreeRegressor(criterion='mse', max_depth=None,
                                 random_state=seed, min_samples_leaf=1,
                                 min_samples_split=2, splitter='best')
    tree_fit = tree.fit(data, label)
    #print(tree.feature_importances_)
    
    Importance = pd.DataFrame({'Importance':tree.feature_importances_*100}, 
                              index=diabetes.feature_names)
    plt.rc('font', size=8)
    Importance.sort_values(by='Importance', axis=0,
                           ascending=True).plot(kind='barh', color='r')
    plt.title('Decision tree')
    plt.xlabel('Feature importance')

# RandomForest
def rf_model(data, label):
    rf = RandomForestRegressor(n_estimators=50, criterion='mse', 
                               max_depth=None, random_state=seed, 
                               min_samples_leaf=1, min_samples_split=2, 
                               min_impurity_split=None)
    rf_fit = rf.fit(data, label)
    #print(rf.feature_importances_)
    Importance = pd.DataFrame({'Importance':rf.feature_importances_*100},
                              index=diabetes.feature_names)
    plt.rc('font', size=8)
    Importance.sort_values(by='Importance', axis=0,
                           ascending=True).plot(kind='barh', color='r')
    plt.title('Random forest')
    plt.xlabel('Feature importance')
    
# AdaBoost
def adabst_model(data, label):
    adabst = AdaBoostRegressor(base_estimator=None, n_estimators=50, 
                               learning_rate=1.0, loss='linear', 
                               random_state=seed)
    adabst_fit = adabst.fit(data, label)
    #print(adabst.feature_importances_)
    Importance = pd.DataFrame({'Importance':adabst.feature_importances_*100}, 
                              index=diabetes.feature_names)
    plt.rc('font', size=8)
    Importance.sort_values(by='Importance', axis=0,
                           ascending=True).plot(kind='barh', color='r')
    plt.title('Adaboost')
    plt.xlabel('Feature importance')

# Xgboost 
def xgb_model(data, label):
    xgb = XGBRegressor(n_estimators=50, random_state=seed)
    xgb_fit = xgb.fit(data, label)
    #print(xgb.feature_importances_)
    Importance = pd.DataFrame({'Importance':xgb.feature_importances_*100},
                              index=diabetes.feature_names)
    Importance.sort_values(by='Importance', axis=0, 
                           ascending=True).plot(kind='barh', color='r')
    plt.title('Xgboost')
    plt.xlabel('Feature importance')
    plt.rc('font', size=8)

# lightGBM    
def lgbm_model(data, label):
    lgbm = LGBMRegressor(n_estimators=50, random_state=seed)
    lgbm_fit = lgbm.fit(data, label)
    #print(lgbm.feature_importances_)
    Importance = pd.DataFrame({'Importance':lgbm.feature_importances_},
                              index=diabetes.feature_names)
    plt.rc('font', size=8)
    Importance.sort_values(by='Importance', axis=0, 
                           ascending=True).plot(kind='barh', color='r')
    plt.title('LightGBM')
    plt.xlabel('Feature importance')

tree_model(x, y)
rf_model(x, y)
adabst_model(x, y)
xgb_model(x, y)
lgbm_model(x, y)

# data preprocess
def prepro(data, method):
    if method == 'minmax':
        scaler = MinMaxScaler()
        x = data.reshape(-1,1)
        scaler.fit(x)
        tmp = scaler.transform(x)
        return np.squeeze(tmp, axis=1)
    
    if method == 'stand':
        scaler = StandardScaler()
        x = data.reshape(-1,1)
        scaler.fit(x)
        tmp = scaler.transform(x)
        return np.squeeze(tmp, axis=1)
    
    if method == 'robust':
        scaler = RobustScaler()
        x = data.reshape(-1,1)
        scaler.fit(x)
        tmp = scaler.transform(x)
        return np.squeeze(tmp, axis=1)
    
    if method == 'maxabs':
        scaler = MaxAbsScaler()
        x = data.reshape(-1,1)
        scaler.fit(x)
        tmp = scaler.transform(x)
        return np.squeeze(tmp, axis=1)

# PCA
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

n = np.argmax(cumsum >= 0.95) + 1
print(n)

def pca_data(data, n):
    pca = PCA(n_components=n)
    x = data.copy()
    w = pca.fit_transform(x)
    return w

x_minmax = np.zeros((x.shape[0], x.shape[1]))
for i in range(len(x[0])):
    x_minmax[:,i] = prepro(x[:,i], 'minmax')

x_stand = np.zeros((x.shape[0], x.shape[1]))
for i in range(len(x[0])):
    x_stand[:,i] = prepro(x[:,i], 'stand')

x_robust = np.zeros((x.shape[0], x.shape[1]))
for i in range(len(x[0])):
    x_robust[:,i] = prepro(x[:,i], 'robust')

x_maxabs = np.zeros((x.shape[0], x.shape[1]))
for i in range(len(x[0])):
    x_maxabs[:,i] = prepro(x[:,i], 'maxabs')

x_pca_7 = pca_data(x, 7)
x_pca_8 = pca_data(x, 8)

print(x_minmax.shape)
print(x_stand.shape)
print(x_robust.shape)
print(x_maxabs.shape)
print(x_pca_7.shape)
print(x_pca_8.shape)

# Grid search / Random search
def DNN(data=x_minmax, drop=0.2, optimizer='adam'):
    x_train = data.shape[1]
    model_input = Input(shape=x_train)
    x = Dense(100, activation='selu')(model_input)
    x = Dropout(drop)(x)
    x = Dense(50, activation='selu')(x)
    x = Dropout(drop)(x)
    x = Dense(10, activation='selu')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=model_input, outputs=outputs)
    model.compile(optimizer = optimizer, metrics = ['mse'],
                  loss = 'mse')
    
    return model

def hyperparameters():
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
    batches = [1, 8, 16, 32]
    epochs = [10, 30, 50]
    optimizers = ['adam', 'adadelta', 'rmsprop']
    return {'drop':dropout, 'batch_size':batches, 'epochs':epochs, 'optimizer':optimizers}
hyper = hyperparameters()

def DNN_pca7(data=x_pca_7, drop=0.2, optimizer='adam'):
    x_train = data.shape[1]
    model_input = Input(shape=x_train)
    #x = Dense(100, activation='selu')(model_input)
    #x = Dropout(drop)(x)
    x = Dense(50, activation='selu')(model_input)
    x = Dropout(drop)(x)
    x = Dense(10, activation='selu')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=model_input, outputs=outputs)
    model.compile(optimizer = optimizer, metrics = ['mse'],
                  loss = 'mse')
    
    return model

def DNN_pca8(data=x_pca_8, drop=0.2, optimizer='adam'):
    x_train = data.shape[1]
    model_input = Input(shape=x_train)
    #x = Dense(100, activation='selu')(model_input)
    #x = Dropout(drop)(x)
    x = Dense(50, activation='selu')(model_input)
    x = Dropout(drop)(x)
    x = Dense(10, activation='selu')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=model_input, outputs=outputs)
    model.compile(optimizer = optimizer, metrics = ['mse'],
                  loss = 'mse')
    
    return model

def best_model(data, label, search_method):
    model = KerasRegressor(build_fn = DNN, verbose=1)
    if search_method == 'grid':
        search = GridSearchCV(model, hyper, cv=5)
        best_search = search.fit(data, label, verbose=2)
        best_param = search.best_params_
        return best_search, best_param
    if search_method == 'random':
        search = RandomizedSearchCV(model, hyper, cv=5)
        best_search = search.fit(data, label, verbose=2)
        best_param = search.best_params_
        return best_search, best_param
    
def best_model_pca7(data, label, search_method):
    model = KerasRegressor(build_fn = DNN_pca7, verbose=1)
    if search_method == 'grid':
        search = GridSearchCV(model, hyper, cv=5)
        best_search = search.fit(data, label, verbose=2)
        best_param = search.best_params_
        return best_search, best_param
    if search_method == 'random':
        search = RandomizedSearchCV(model, hyper, cv=5)
        best_search = search.fit(data, label, verbose=2)
        best_param = search.best_params_
        return best_search, best_param
    
def best_model_pca8(data, label, search_method):
    model = KerasRegressor(build_fn = DNN_pca8, verbose=1)
    if search_method == 'grid':
        search = GridSearchCV(model, hyper, cv=5)
        best_search = search.fit(data, label, verbose=2)
        best_param = search.best_params_
        return best_search, best_param
    if search_method == 'random':
        search = RandomizedSearchCV(model, hyper, cv=5)
        best_search = search.fit(data, label, verbose=2)
        best_param = search.best_params_
        return best_search, best_param

minmax_model, minmax_param = best_model(x_minmax, y, 'random')
stand_model, stand_param = best_model(x_stand, y, 'random')
robust_model, robust_param = best_model(x_robust, y, 'random')
maxabs_model, maxabs_param = best_model(x_maxabs, y, 'random')
pca7_model, pca7_param = best_model_pca7(x_pca_7, y, 'random')
pca8_model, pca8_param = best_model_pca8(x_pca_8, y, 'random')

# hyperparameter tuning result
print(minmax_model,'\n', minmax_param)
print(stand_model,'\n', stand_param)
print(robust_model,'\n', robust_param)
print(maxabs_model,'\n', maxabs_param)
print(pca7_model,'\n', pca7_param)
print(pca8_model,'\n', pca8_param)

# prediction
x_test_minmax = x_minmax[:50,:]
x_test_stand = x_stand[:50,:]
x_test_robust = x_robust[:50,:]
x_test_maxabs = x_maxabs[:50,:]
x_test_pca7 = x_pca_7[:50,:]
x_test_pca8 = x_pca_8[:50,:]

minmax_pred = minmax_model.predict(x_test_minmax)
stand_pred = stand_model.predict(x_test_stand)
robust_pred = robust_model.predict(x_test_robust)
maxabs_pred = maxabs_model.predict(x_test_maxabs)
pca7_pred = pca7_model.predict(x_test_pca7)
pca8_pred = pca8_model.predict(x_test_pca8)

x = np.arange(0,50,1)
y_test = y[:50]

plt.figure(figsize=(12,10))
plt.subplot(3,2,1)
plt.title('MinMax')
plt.plot(x,y_test,'r',label='ture')
plt.plot(x,minmax_pred,'b',label='predict')
plt.legend(loc='upper right')

plt.subplot(3,2,2)
plt.title('Standard')
plt.plot(x,y_test,'r',label='ture')
plt.plot(x,stand_pred,'b',label='predict')
plt.legend(loc='upper right')

plt.subplot(3,2,3)
plt.title('Robust')
plt.plot(x,y_test,'r',label='ture')
plt.plot(x,robust_pred,'b',label='predict')
plt.legend(loc='upper right')

plt.subplot(3,2,4)
plt.title('MaxAbs')
plt.plot(x,y_test,'r',label='ture')
plt.plot(x,maxabs_pred,'b',label='predict')
plt.legend(loc='upper right')

plt.subplot(3,2,5)
plt.title('PCA7')
plt.plot(x,y_test,'r',label='ture')
plt.plot(x,pca7_pred,'b',label='predict')
plt.legend(loc='upper right')

plt.subplot(3,2,6)
plt.title('PCA8')
plt.plot(x,y_test,'r',label='ture')
plt.plot(x,pca8_pred,'b',label='predict')
plt.legend(loc='upper right')