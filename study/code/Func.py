import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


class TreeModel:

    def __init__(self, data, label, columns_name, seed):
        self.data = data
        self.label = label
        self.columns_name = columns_name
        self.seed = seed

    # Decision Tree
    def tree_model(self):
        tree = DecisionTreeRegressor(criterion='mse', max_depth=None,
                                    random_state=self.seed, min_samples_leaf=1,
                                    min_samples_split=2, splitter='best')
        tree_fit = tree.fit(self.data, self.label)
        Importance = pd.DataFrame({'Importance':tree.feature_importances_*100}, 
                                index=self.columns_name)
        plt.rc('font', size=8)
        Importance.sort_values(by='Importance', axis=0,
                            ascending=True).plot(kind='barh', color='r')
        plt.title('Decision tree')
        plt.xlabel('Feature importance')

    # RandomForest
    def rf_model(self):
        rf = RandomForestRegressor(n_estimators=50, criterion='mse', 
                                max_depth=None, random_state=self.seed, 
                                min_samples_leaf=1, min_samples_split=2, 
                                min_impurity_split=None)
        rf_fit = rf.fit(self.data, self.label)
        Importance = pd.DataFrame({'Importance':rf.feature_importances_*100},
                                index=self.columns_name)
        plt.rc('font', size=8)
        Importance.sort_values(by='Importance', axis=0,
                            ascending=True).plot(kind='barh', color='r')
        plt.title('Random forest')
        plt.xlabel('Feature importance')
        
    # AdaBoost
    def adabst_model(self):
        adabst = AdaBoostRegressor(base_estimator=None, n_estimators=50, 
                                learning_rate=1.0, loss='linear', 
                                random_state=self.seed)
        adabst_fit = adabst.fit(self.data, self.label)
        Importance = pd.DataFrame({'Importance':adabst.feature_importances_*100}, 
                                index=self.columns_name)
        plt.rc('font', size=8)
        Importance.sort_values(by='Importance', axis=0,
                            ascending=True).plot(kind='barh', color='r')
        plt.title('Adaboost')
        plt.xlabel('Feature importance')
        
    # Xgboost
    def xgb_model(self):
        xgb = XGBRegressor(n_estimators=50, random_state=self.seed)
        xgb_fit = xgb.fit(self.data, self.label)
        Importance = pd.DataFrame({'Importance':xgb.feature_importances_*100},
                                index=self.columns_name)
        Importance.sort_values(by='Importance', axis=0, 
                            ascending=True).plot(kind='barh', color='r')
        plt.title('Xgboost')
        plt.xlabel('Feature importance')
        plt.rc('font', size=8)
        
    # lightGBM
    def lgbm_model(self):
        lgbm = LGBMRegressor(n_estimators=50, random_state=self.seed)
        lgbm_fit = lgbm.fit(self.data, self.label)
        Importance = pd.DataFrame({'Importance':lgbm.feature_importances_},
                                index=self.columns_name)
        plt.rc('font', size=8)
        Importance.sort_values(by='Importance', axis=0, 
                            ascending=True).plot(kind='barh', color='r')
        plt.title('LightGBM')
        plt.xlabel('Feature importance')

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

def pca_data(data, n):
    pca = PCA(n_components=n)
    x = data.copy()
    w = pca.fit_transform(x)
    return w

class BestModel:

    def __init__(self, data, label, search_method):
        self.data = data
        self.label = label
        self.search_method = search_method

    def DNN(self, drop, optimizer):
        x_train = self.data.shape[1]
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

    def DNN_pca2(self, drop, optimizer):
        x_train = self.data.shape[1]
        model_input = Input(shape=x_train)
        x = Dense(30, activation='selu')(model_input)
        x = Dropout(drop)(x)
        x = Dense(10, activation='selu')(x)
        x = Dropout(drop)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=model_input, outputs=outputs)
        model.compile(optimizer = optimizer, metrics = ['mse'],
                    loss = 'mse')
        
        return model

    def DNN_pca3(self, drop, optimizer):
        x_train = self.data.shape[1]
        model_input = Input(shape=x_train)
        x = Dense(30, activation='selu')(model_input)
        x = Dropout(drop)(x)
        x = Dense(10, activation='selu')(x)
        x = Dropout(drop)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=model_input, outputs=outputs)
        model.compile(optimizer = optimizer, metrics = ['mse'],
                    loss = 'mse')
        
        return model
    
    def DNN_pca7(self, drop, optimizer):
        x_train = self.data.shape[1]
        model_input = Input(shape=x_train)
        x = Dense(50, activation='selu')(model_input)
        x = Dropout(drop)(x)
        x = Dense(10, activation='selu')(x)
        x = Dropout(drop)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=model_input, outputs=outputs)
        model.compile(optimizer = optimizer, metrics = ['mse'],
                    loss = 'mse')
        
        return model

    def DNN_pca8(self, drop, optimizer):
        x_train = self.data.shape[1]
        model_input = Input(shape=x_train)
        x = Dense(50, activation='selu')(model_input)
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
        return {'drop':dropout, 'batch_size':batches, 'epochs':epochs, 
                'optimizer':optimizers}
    hyper = hyperparameters()

    def best_model(self):
        model = KerasRegressor(build_fn = self.DNN, verbose=1)
        if self.search_method == 'grid':
            search = GridSearchCV(model, self.hyper, cv=5)
            best_search = search.fit(self.data, self.label, verbose=2)
            best_param = search.best_params_
            return best_search, best_param
        if self.search_method == 'random':
            search = RandomizedSearchCV(model, self.hyper, cv=5)
            best_search = search.fit(self.data, self.label, verbose=2)
            best_param = search.best_params_
            return best_search, best_param
    
    def best_model_pca2(self):
        model = KerasRegressor(build_fn = self.DNN_pca2, verbose=1)
        if self.search_method == 'grid':
            search = GridSearchCV(model, self.hyper, cv=5)
            best_search = search.fit(self.data, self.label, verbose=2)
            best_param = search.best_params_
            return best_search, best_param
        if self.search_method == 'random':
            search = RandomizedSearchCV(model, self.hyper, cv=5)
            best_search = search.fit(self.data, self.label, verbose=2)
            best_param = search.best_params_
            return best_search, best_param
    
    def best_model_pca3(self):
        model = KerasRegressor(build_fn = self.DNN_pca3, verbose=1)
        if self.search_method == 'grid':
            search = GridSearchCV(model, self.hyper, cv=5)
            best_search = search.fit(self.data, self.label, verbose=2)
            best_param = search.best_params_
            return best_search, best_param
        if self.search_method == 'random':
            search = RandomizedSearchCV(model, self.hyper, cv=5)
            best_search = search.fit(self.data, self.label, verbose=2)
            best_param = search.best_params_
            return best_search, best_param

    def best_model_pca7(self):
        model = KerasRegressor(build_fn = self.DNN_pca7, verbose=1)
        if self.search_method == 'grid':
            search = GridSearchCV(model, self.hyper, cv=5)
            best_search = search.fit(self.data, self.label, verbose=2)
            best_param = search.best_params_
            return best_search, best_param
        if self.search_method == 'random':
            search = RandomizedSearchCV(model, self.hyper, cv=5)
            best_search = search.fit(self.data, self.label, verbose=2)
            best_param = search.best_params_
            return best_search, best_param
    
    def best_model_pca8(self):
        model = KerasRegressor(build_fn = self.DNN_pca8, verbose=1)
        if self.search_method == 'grid':
            search = GridSearchCV(model, self.hyper, cv=5)
            best_search = search.fit(self.data, self.label, verbose=2)
            best_param = search.best_params_
            return best_search, best_param
        if self.search_method == 'random':
            search = RandomizedSearchCV(model, self.hyper, cv=5)
            best_search = search.fit(self.data, self.label, verbose=2)
            best_param = search.best_params_
            return best_search, best_param