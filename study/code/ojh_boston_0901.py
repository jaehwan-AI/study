# import library
from sklearn.datasets import load_boston
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
boston = load_boston()

# dataframe
data = pd.DataFrame(boston.data, columns=boston.feature_names)
target = pd.DataFrame(boston.target, columns=["MEDV"])
df = pd.concat([data, target], axis=1)
df

df.describe()
'''
# CRIM = 범죄율 / ZN = 25,000 평방피트 초과 거주지역 비율
# INDUS = 비소매상업지역 면적 비율 / CHAS = 찰스강의 경계 위치 여부(위치=1,아니면=0)
# NOX = 일산화질소 농도 / RM = 주택당 방 수 / AGE = 1940년 이전에 건축된 주택의 비율
# DIS = 직업센터의 거리 / RAD = 방사형 고속도로까지의 거리 / TAX = 재산세율
# PTRATIO = 학생,교사 비율 / B = 인구 중 흑인 비율 / LSTAT = 인구 중 하위 계층 비율
# MEDV = 1978년 보스턴 주택 가격, 506개 타운의 주택 가격 중앙값(단위:1,000달러)
'''

# 데이터 분포
plt.figure(figsize=(30,30))
for i in range(len(df.columns)):
    plt.rc('font', size=25)
    plt.subplot(5,3,i+1)
    df[df.columns[i]].plot.hist()
    plt.title(f'{df.columns[i]}')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
'''
# CHAS -> binary data(이진 데이터)
# CHAS를 제외한 나머지 데이터 -> continuous data(연속형 데이터)
'''

# dataset
x = boston.data
y = boston.target

# Decision Tree
def tree_model(data, label):
    tree = DecisionTreeRegressor(criterion='mse', max_depth=None,
                                 random_state=seed, min_samples_leaf=1,
                                 min_samples_split=2, splitter='best')
    tree_fit = tree.fit(data, label)
    #print(tree.feature_importances_)
    
    Importance = pd.DataFrame({'Importance':tree.feature_importances_*100}, 
                              index=boston.feature_names)
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
                              index=boston.feature_names)
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
                              index=boston.feature_names)
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
                              index=boston.feature_names)
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
                              index=boston.feature_names)
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

x_pca_2 = pca_data(x, 2)
x_pca_3 = pca_data(x, 3)

print(x_minmax.shape)
print(x_stand.shape)
print(x_robust.shape)
print(x_maxabs.shape)
print(x_pca_2.shape)
print(x_pca_3.shape)

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

def DNN_pca2(data=x_pca_2, drop=0.2, optimizer='adam'):
    x_train = data.shape[1]
    model_input = Input(shape=x_train)
    #x = Dense(100, activation='selu')(model_input)
    #x = Dropout(drop)(x)
    x = Dense(30, activation='selu')(model_input)
    x = Dropout(drop)(x)
    x = Dense(10, activation='selu')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=model_input, outputs=outputs)
    model.compile(optimizer = optimizer, metrics = ['mse'],
                  loss = 'mse')
    
    return model

def DNN_pca3(data=x_pca_3, drop=0.2, optimizer='adam'):
    x_train = data.shape[1]
    model_input = Input(shape=x_train)
    #x = Dense(100, activation='selu')(model_input)
    #x = Dropout(drop)(x)
    x = Dense(30, activation='selu')(model_input)
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
    
def best_model_pca2(data, label, search_method):
    model = KerasRegressor(build_fn = DNN_pca2, verbose=1)
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
    
def best_model_pca3(data, label, search_method):
    model = KerasRegressor(build_fn = DNN_pca3, verbose=1)
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
pca2_model, pca2_param = best_model_pca2(x_pca_2, y, 'random')
pca3_model, pca3_param = best_model_pca3(x_pca_3, y, 'random')

# hyperparameter tuning result
print(minmax_model,'\n', minmax_param)
print(stand_model,'\n', stand_param)
print(robust_model,'\n', robust_param)
print(maxabs_model,'\n', maxabs_param)
print(pca2_model,'\n', pca2_param)
print(pca3_model,'\n', pca3_param)

# prediction
x_test_minmax = x_minmax[:50,:]
x_test_stand = x_stand[:50,:]
x_test_robust = x_robust[:50,:]
x_test_maxabs = x_maxabs[:50,:]
x_test_pca2 = x_pca_2[:50,:]
x_test_pca3 = x_pca_3[:50,:]

minmax_pred = minmax_model.predict(x_test_minmax)
stand_pred = stand_model.predict(x_test_stand)
robust_pred = robust_model.predict(x_test_robust)
maxabs_pred = maxabs_model.predict(x_test_maxabs)
pca2_pred = pca2_model.predict(x_test_pca2)
pca3_pred = pca3_model.predict(x_test_pca3)

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
plt.title('PCA2')
plt.plot(x,y_test,'r',label='ture')
plt.plot(x,pca2_pred,'b',label='predict')
plt.legend(loc='upper right')

plt.subplot(3,2,6)
plt.title('PCA3')
plt.plot(x,y_test,'r',label='ture')
plt.plot(x,pca3_pred,'b',label='predict')
plt.legend(loc='upper right')