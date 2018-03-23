from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LassoLarsCV

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        
        X = X.values
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X.values) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
        def __init__(self, models):
            self.models = models
            
        # we define clones of the original models to fit the data in
        def fit(self, X, y):
            self.models_ = [clone(x) for x in self.models]
            
            # Train cloned base models
            for model in self.models_:
                model.fit(X, y)
    
            return self
        
        #Now we do the predictions for cloned models and average them
        def predict(self, X):
            predictions = np.column_stack([
                model.predict(X) for model in self.models_
            ])
            return np.mean(predictions, axis=1) 
        
class EnsembleModels():
        
    def __init__(self, models, weighting = [0.7,0.15,0.15]):
        self.models = models
        self.weighting = weighting
    
    
    def fit(self, X, y):
        
        self.models_ = [clone(x) for x in self.models]
            
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
    
        return self 
    
    def predict(self, X):
        
        predictions = np.column_stack([
                model.predict(X) for model in self.models_
            ])
    
        predictions = np.expm1(predictions)
        predictions = np.dot(predictions, self.weighting)
    
        return predictions
        
        

class models():
    
    def __init__(self, train, y_train, test, n_folds):
        self.train = train
        self.y_train = y_train
        self.test = test
        self.n_folds = n_folds #Validation function
        
        
    def rmsle_cv(self, model):
        kf = KFold(self.n_folds, shuffle=True, random_state=42).get_n_splits(self.train.values)
        rmse= np.sqrt(-cross_val_score(model, self.train.values, self.y_train, scoring="neg_mean_squared_error", cv = kf))
        return(rmse)
        
    def rmsle(self, y, y_pred):
        return np.sqrt(mean_squared_error(y, y_pred))

  
    
    def fit_model_and_submission(self, model, model_name, exp = True):
        model.fit(self.train, self.y_train)
        pred_train = model.predict(self.train)
        #pred_test = model.predict(self.test)
       
        
        if exp:
             rmse = self.rmsle(self.y_train, pred_train)
             print("rmse_",model_name," : ", rmse)
            
        else:
             rmse = self.rmsle(self.y_train, np.log1p(pred_train))
             print("rmse_",model_name," : ", rmse)
           
             
    
    def model_config1(self):
        #LASSO Regression 
        self.lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
        #Elastic Net Regression
        self.ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
        #Kernel Ridge Regression
        self.KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
        #Gradient Boosting Regression
        self.Boost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                           max_depth=4, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=10, 
                                           loss='huber', random_state =5)
        #XGBoost
        self.model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                                     learning_rate=0.05, max_depth=3, 
                                     min_child_weight=1.7817, n_estimators=2200,
                                     reg_alpha=0.4640, reg_lambda=0.8571,
                                     subsample=0.5213, silent=1,
                                     random_state =7, nthread = -1)
        
        #LightGBM
        self.model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                                      learning_rate=0.05, n_estimators=720,
                                      max_bin = 55, bagging_fraction = 0.8,
                                      bagging_freq = 5, feature_fraction = 0.2319,
                                      feature_fraction_seed=9, bagging_seed=9,
                                      min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
        
        self.average_model = AveragingModels(models = (self.ENet, self.Boost, self.KRR))
        
        self.stack_model = StackingAveragedModels(base_models = ([self.ENet, self.Boost, self.KRR]),
                                                 meta_model = self.lasso)
        
        self.ensemble_model = EnsembleModels([self.stack_model, self.model_xgb, self.model_lgb], [0.7, 0.15, 0.15])
        
        
    def model_config2(self):  
        
        self.model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])
        self.model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)
        self.ensemble_model = EnsembleModels([self.model_lasso, self.model_xgb], [0.7, 0.3])
         
   
        
        
        
    