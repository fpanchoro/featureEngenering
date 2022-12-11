#install
#!pip install --upgrade category_encoders
#!pip install boruta
#!pip install borutashap
#!pip3 install catboost
#!pip install eif
#!pip install h2o

# Import Libraries

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import random
import time
import pandas_profiling as pp
import glob
import re

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.stats import pointbiserialr,chi2_contingency
import category_encoders as ce

import sklearn
from boruta import BorutaPy
from sklearn.linear_model import LogisticRegression, LassoCV
import statsmodels.api as sm
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from BorutaShap import BorutaShap
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV



from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import statsmodels.api as sm

from sklearn.ensemble import IsolationForest
import eif as iso
import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator
from sklearn.covariance import EllipticEnvelope


from scipy.stats import norm
from sklearn.metrics import (roc_auc_score,classification_report,confusion_matrix,
                             precision_recall_curve, auc, roc_curve, recall_score, f1_score, 
                             precision_recall_fscore_support, roc_auc_score)
 


import seaborn as sns
import matplotlib.pyplot as plt

class FeatureEngenering:
 
    def __init__(self):
        print("calling __init__() constructor...")
   
   # Data Prep Functions
    def data_prep(self,data, fields_to_drop, string_columns, numeric_columns,null_handling_required):
        try:
            data_v2 = data.copy()
            #Drop unwanted fields
            if len(fields_to_drop)>0:
                data_v2 = data_v2.drop(fields_to_drop, axis = 1)
                print('Dropped {} Fields'.format(len(fields_to_drop)))
            else:
                print('No Fields to Drop')
                
            # Convert String Columns
            
            if len(string_columns)>0:
                data_v2[string_columns] = data_v2[string_columns].astype(str)
                print('{} Fields converted to string'.format(len(string_columns)))
            else:
                print('No Fields to Convert to String')
                
                
            # Convert NUmeric Columns
            
            if len(numeric_columns)>0:
                data_v2[numeric_columns] = data_v2[string_columns].astype(float)
                print('{} Fields converted to string'.format(len(numeric_columns)))
            else:
                print('No Fields to Convert to Numeric')
                
            #Null Treatment
            
            if null_handling_required == True:
                print('Number of records before dropping nulls:',len(data_v2))
                data_v2 = data_v2.dropna()
                print('Number of records after dropping nulls:',len(data_v2))
            else:
                print('No Null Treatment')
                
        except:
            print('Error: Reload data and check manual inputs')
            
        return data_v2




    # WOE
    def WOE(self, train_features, test_features,train_target, test_target):
        WOE_columns = train_features.select_dtypes(exclude = [np.number]).columns
        
        woe_encoder = ce.WOEEncoder(cols = WOE_columns)
        
        WOE_encoded_train = woe_encoder.fit_transform(train_features[WOE_columns],train_target).add_suffix('_woe')
        train_features = pd.concat([train_features,WOE_encoded_train],axis=1)
        
        WOE_encoded_test = woe_encoder.transform(test_features[WOE_columns],test_target).add_suffix('_woe')
        test_features = pd.concat([test_features,WOE_encoded_test],axis=1)
        
        train_features_v2 = train_features.drop(WOE_columns,axis =1)
        test_features_v2 = test_features.drop(WOE_columns,axis =1)
        
        return train_features_v2, test_features_v2
        
                
                
    # Train Test Prep Function
    def train_test(self, data, target, test, shuffle_flag, stratify_flag,WOE_encoding):
        
        # Define X and Y
        features = data.drop([target], axis=1)
        Target = data[target]
        print('Target Distribution:')
        print(data[target].value_counts())
        
        
        # Treating Categorical Features and Train/Test Split
        
        
        if WOE_encoding == True:
            # Train and Test Split
            if stratify_flag == True:
                train_features, test_features, train_target, test_target = train_test_split(features, Target, test_size = test, shuffle = shuffle_flag, stratify = Target)
            else:
                train_features, test_features, train_target, test_target = train_test_split(features, Target, test_size = test, shuffle = shuffle_flag)
                
            train_features, test_features = self.WOE(train_features, test_features,train_target, test_target)
            
        else:
            features = pd.get_dummies(features, drop_first = True)
            # Train and Test Split
            
            if stratify_flag == True:
                train_features, test_features, train_target, test_target = train_test_split(features, Target, test_size = test, shuffle = shuffle_flag, stratify = Target)
            else:
                train_features, test_features, train_target, test_target = train_test_split(features, Target, test_size = test, shuffle = shuffle_flag)
        
        # Summary
        
        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', train_target.shape)
        
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', test_target.shape)
        
        train_features = train_features.astype(float)
        train_target = train_target.astype(float)
        test_features = test_features.astype(float)
        test_target = test_target.astype(float)
        
        return train_features, test_features, train_target, test_target, features, Target
        
 
    #1.Select the top n features based on absolute correlation with train_target variable
    # Function to calculate Cramer's V
    def cramers_V(self,var1,var2) :
        crosstab=np.array(pd.crosstab(var1,var2, 
                                        rownames=None, colnames=None))
        stat = chi2_contingency(crosstab)[0]
        obs = np.sum(crosstab) 
        mini = min(crosstab.shape)-1 
        return (stat/(obs*mini))


    # Overall Correlation Function
    def corr_feature_selection(self,data,target,pearson_list,
                            point_bi_serial_list,cramer_list,
                            pearson_threshold,
                            point_bi_serial_threshold,
                            cramer_threshold):
        
        #Inputs
        # data - Input feature data
        # target - Target Variable
        # pearson_list - list of continuous features (if target is continuous)
        # point_bi_serial_list - list of continuous features (if target is categorical)/
        #                        list of categorical features (if target is continuous)   
        # cramer_list - list of categorical features (if target is categorical)
        # pearson_threshold - select features if pearson corrrelation is above this
        # point_bi_serial_threshold - select features if biserial corrrelation is above this
        # cramer_threshold - select features if cramer's v is above this  
        
        corr_data = pd.DataFrame()

        # Calculate point bi-serial
        for i in point_bi_serial_list:
            # Manual Change in Parameters - Point Bi-Serial
            pbc = pointbiserialr(target, data[i])   
            corr_temp_data = [[i,pbc.correlation,"point_bi_serial"]]
            corr_temp_df = pd.DataFrame(corr_temp_data, 
                                        columns = ['Feature', 
                                                'Correlation',
                                                'Correlation_Type'])
            corr_data = corr_data.append(corr_temp_df)

        # Calculate cramer's v
        for i in cramer_list:
            cramer = cramers_V(target, data[i])
            corr_temp_data = [[i,cramer,"cramer_v"]]
            corr_temp_df = pd.DataFrame(corr_temp_data,
                                        columns = ['Feature',
                                                'Correlation',
                                                'Correlation_Type'])
            corr_data = corr_data.append(corr_temp_df)


        # Calculate pearson correlation
        for i in pearson_list:
            # Manual Change in Parameters - Perason
            pearson = target.corr(data[i])
            corr_temp_data = [[i,pearson,"pearson"]]
            corr_temp_df = pd.DataFrame(corr_temp_data,
                                        columns = ['Feature',
                                                'Correlation',
                                                'Correlation_Type'])
            corr_data = corr_data.append(corr_temp_df)


        # Filter NA and sort based on absolute correlation
        corr_data = corr_data.iloc[corr_data.Correlation.abs().argsort()] 
        corr_data = corr_data[corr_data['Correlation'].notna()]
        corr_data = corr_data.loc[corr_data['Correlation'] != 1]
        
        # Add thresholds
        
        # initialize list of lists
        data = [['pearson', pearson_threshold],
                ['point_bi_serial', point_bi_serial_threshold],
                ['cramer_v', cramer_threshold]]

        threshold_df = pd.DataFrame(data,
                                    columns=['Correlation_Type',
                                            'Threshold'])
        corr_data = pd.merge(corr_data,threshold_df,
                            on=['Correlation_Type'],how = 'left')
        



        # Select Features with greater than user dfined absolute correlation
        corr_data2 = corr_data.loc[corr_data['Correlation'].abs() > corr_data['Threshold']]
        corr_top_features = corr_data2['Feature'].tolist()
        print(corr_top_features)
        corr_top_features_df = pd.DataFrame(corr_top_features,columns = ['Feature'])
        corr_top_features_df['Method'] = 'Correlation'
        return corr_data,corr_top_features_df

    #2. Select top features based on information value

    def iv_woe(self, data, target, iv_bins,iv_threshold, show_woe):
        
        #Inputs
        # data - Input Data including target variable
        # target - Target Variable name
        # iv_bins - Number of iv_bins
        # show_woe - show all the iv_bins and features
        # iv_threshold - select features with IV greater than this
        
        #Empty Dataframe
        newDF,woeDF = pd.DataFrame(), pd.DataFrame()
        
        #Extract Column Names
        cols = data.columns
        
        #Run WOE and IV on all the independent variables
        for ivars in cols[~cols.isin([target])]:
            if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
                binned_x = pd.qcut(data[ivars], iv_bins,  duplicates='drop')
                d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
            else:
                d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})

            
            # Calculate the number of events in each group (bin)
            d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
            d.columns = ['Cutoff', 'N', 'Events']
            
            # Calculate % of events in each group.
            d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()

            # Calculate the non events in each group.
            d['Non-Events'] = d['N'] - d['Events']
            # Calculate % of non events in each group.
            d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()

            # Calculate WOE by taking natural log of division of % 
            # of non-events and % of events
            d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
            d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
            d.insert(loc=0, column='Variable', value=ivars)
            print("Information value of " + ivars + " is " + 
                str(round(d['IV'].sum(),6)))
            temp =pd.DataFrame({"Variable" : [ivars],
                                "IV" : [d['IV'].sum()]},
                            columns = ["Variable", "IV"])
            newDF=pd.concat([newDF,temp], axis=0)
            woeDF=pd.concat([woeDF,d], axis=0)

            #Show WOE Table
            if show_woe == True:
                print(d)
        
        # Aggregate IV at feature level
        woeDF_v2 = pd.DataFrame(woeDF.groupby('Variable')['IV'].agg('sum'),
                                columns= ['IV']).reset_index()
        woeDF_v3 = woeDF_v2.sort_values(['IV'], ascending = False)
        IV_df = woeDF_v2[woeDF_v2['IV']> iv_threshold]
        woe_top_features = IV_df['Variable'].tolist()
        print(woe_top_features)
        woe_top_features_df = pd.DataFrame(woe_top_features,columns = ['Feature'])
        woe_top_features_df['Method'] = 'Information_value'
        return newDF, woeDF,IV_df, woe_top_features_df

    #3. Select the top n features based on absolute value of beta coefficient of features

    def beta_coeff(self, data, train_target,beta_threshold):
        
        #Inputs
        # data - Input feature data 
        # train_target - Target variable training data
        # beta_threshold - select n features with highest absolute beta coeficient value
        
        # Standardise dataset

        scaler = StandardScaler()
        data_v2 = pd.DataFrame(scaler.fit_transform(data))
        data_v2.columns = data.columns

        # Fit Logistic on Standardised dataset
        # Manual Change in Parameters - Logistic Regression      
        log = LogisticRegression(fit_intercept = False, penalty = 'none')
        log.fit(data_v2, train_target)
        coef_table = pd.DataFrame(list(data_v2.columns)).copy()
        coef_table.insert(len(coef_table.columns), "Coefs", log.coef_.transpose())
        coef_table = coef_table.iloc[coef_table.Coefs.abs().argsort()]
        sr_data2 = coef_table.tail(beta_threshold)
        beta_top_features = sr_data2.iloc[:,0].tolist()
        print(beta_top_features)
        
        beta_top_features_df = pd.DataFrame(beta_top_features,columns = ['Feature'])
        beta_top_features_df['Method'] = 'Beta_coefficients'

        log_v2 = sm.Logit(train_target, sm.add_constant(data[beta_top_features])).fit()
        print('Logistic Regression with selected features')
        print(log_v2.summary())
        
        return log,log_v2,beta_top_features_df

    #4. Select the features identified by Lasso regression

    def lasso(self, data, train_target,lasso_param):
        
        #Inputs
        # data - Input feature data 
        # train_target - Target variable training data
        # lasso_param - Lasso l1 penalty term
        
        #Fit Logistic
        # Manual Change in Parameters - Logistic Regression      
        log = LogisticRegression(penalty ='l1', solver = 'liblinear', C = lasso_param)
        log.fit(data, train_target)
        
        #Select Features
        lasso_df = pd.DataFrame(columns = ['Feature', 'Lasso_Coef'])
        lasso_df['Feature'] = data.columns
        lasso_df['Lasso_Coef'] = log.coef_.squeeze().tolist()
        lasso_df_v2 = lasso_df[lasso_df['Lasso_Coef'] !=0]
        lasso_top_features = lasso_df_v2['Feature'].tolist()
        
        lasso_top_features_df = pd.DataFrame(lasso_top_features,columns = ['Feature'])
        lasso_top_features_df['Method'] = 'Lasso'


        # Logistic Regression with selected features
        log_v2 = sm.Logit(train_target, sm.add_constant(data[lasso_top_features])).fit()
        print('Logistic Regression with selected features')
        print(log_v2.summary())
        
        return log_v2,lasso_top_features_df

    #5. Select features based on Recursive Feature Selection method

    def rfecv_feature_selection(self, data, train_target,rfe_estimator,rfe_step,rfe_cv,rfe_scoring):
        
        #Inputs
        # data - Input feature data 
        # train_target - Target variable training data
        # rfe_estimator - base model (default: Decision Tree)
        # rfe_step -  number of features to remove at each iteration
        # rfe_cv - cross-validation splitting strategy
        # rfe_scoring - CV performance scoring metric
        

        ## Initialize RFE

        if rfe_estimator == "XGBoost":
            # Manual Change in Parameters - XGBoost
            # Link to function parameters - https://xgboost.readthedocs.io/en/stable/parameter.html       
            estimator_rfe = XGBClassifier(n_jobs = -1, random_state=101, eval_metric='mlogloss' )
        elif rfe_estimator == "RandomForest":
            # Manual Change in Parameters - RandomForest
            # Link to function parameters - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            estimator_rfe = RandomForestClassifier(n_jobs = -1, random_state=101, eval_metric='mlogloss')
        elif rfe_estimator == "CatBoost":
            # Manual Change in Parameters - CatBoost
            # Link to function parameters - https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier
            estimator_rfe = CatBoostClassifier(iterations=50,verbose=0,random_state=101, eval_metric='mlogloss')
        elif rfe_estimator == "LightGBM":
            # Manual Change in Parameters - LightGBM
            # Link to function parameters - https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
            estimator_rfe = lgb.LGBMClassifier(n_jobs = -1, random_state=101, eval_metric='mlogloss')
        else:
            # Manual Change in Parameters - DecisionTree
            # Link to function parameters - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
            estimator_rfe = DecisionTreeClassifier(random_state=101, eval_metric='mlogloss')


        # Fit RFECV
        # Manual Change in Parameters - RFECV
        # Link to function parameters - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
        # Scoring metrics - https://scikit-learn.org/stable/modules/model_evaluation.html
        rfecv = RFECV(estimator = estimator_rfe, step = rfe_step, cv = rfe_cv, scoring = rfe_scoring)
        rfecv.fit(data, train_target)

        # Select feature based on RFE
        print('Optimal number of features: {}'.format(rfecv.n_features_))
        rfe_df = pd.DataFrame(columns = ['Feature', 'rfe_filter'])
        rfe_df['Feature'] = data.columns
        rfe_df['rfe_filter'] = rfecv.support_.tolist()
        rfe_df_v2 = rfe_df[rfe_df['rfe_filter']==True]
        rfe_top_features = rfe_df_v2['Feature'].tolist()
        print(rfe_top_features)
        
        rfe_top_features_df = pd.DataFrame(rfe_top_features,columns = ['Feature'])
        rfe_top_features_df['Method'] = 'RFECV'

        # Plot CV results
        plt.figure(figsize=(16, 9))
        plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
        plt.ylabel('f1 acore', fontsize=14, labelpad=20)
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

        plt.show()
        
        return rfe_top_features_df,rfecv
    #6. Select features based on Sequential Feature Selector

    def sfs_feature_selection(self, data, train_target,sfs_feature,sfs_direction,sfs_cv,sfs_scoring):
        
        #Inputs
        # data - Input feature data 
        # train_target - Target variable training data
        # sfs_feature - no. of features to select
        # sfs_direction -  forward and backward selection
        # sfs_cv - cross-validation splitting strategy
        # sfs_scoring - CV performance scoring metric
        

        logistic = LogisticRegression(penalty = None)

        sfs = SequentialFeatureSelector(estimator = logistic, n_features_to_select = sfs_feature, direction = sfs_direction,cv = sfs_cv, scoring = sfs_scoring)
        sfs.fit(data, train_target)
        sfs.get_support()

        sfs_df = pd.DataFrame(columns = ['Feature', 'SFS_filter'])
        sfs_df['Feature'] = data.columns
        sfs_df['SFS_filter'] = sfs.get_support().tolist()

        sfs_df_v2 = sfs_df[sfs_df['SFS_filter']==True]
        sfs_top_features = sfs_df_v2['Feature'].tolist()
        print(sfs_top_features)

        log_v2 = sm.Logit(train_target, sm.add_constant(data[sfs_top_features])).fit()
        print(log_v2.summary())
        
        sfs_top_features_df = pd.DataFrame(sfs_top_features,columns = ['Feature'])
        sfs_top_features_df['Method'] = 'Sequential_feature_selector'

        
        return sfs_top_features_df,sfs

    #7. Select features based on BorutaPy method

    def borutapy_feature_selection(self, data, train_target,borutapy_estimator,borutapy_trials,borutapy_green_blue):
        
        #Inputs
        # data - Input feature data 
        # train_target - Target variable training data
        # borutapy_estimator - base model (default: XG Boost)
        # borutapy_trials -  number of iteration
        # borutapy_green_blue - choice for green and blue features
        

        ## Initialize borutapy
        
        if borutapy_estimator == "RandomForest":
            # Manual Change in Parameters - RandomForest
            estimator_borutapy = RandomForestClassifier(n_jobs = -1, random_state=101,max_depth=7)
        elif borutapy_estimator == "LightGBM":
            # Manual Change in Parameters - LightGBM
            estimator_borutapy = lgb.LGBMClassifier(n_jobs = -1, random_state=101,max_depth=7)
        else:
            # Manual Change in Parameters - XGBoost
            estimator_borutapy = XGBClassifier(n_jobs = -1, random_state=101,max_depth=7, eval_metric='mlogloss')

        ## fit Borutapy
        # Manual Change in Parameters - Borutapy
        borutapy = BorutaPy(estimator = estimator_borutapy, n_estimators = 'auto', max_iter = borutapy_trials)
        borutapy.fit(np.array(data), np.array(train_target))
        
        ## print results
        green_area = data.columns[borutapy.support_].to_list()
        blue_area = data.columns[borutapy.support_weak_].to_list()
        print('features in the green area:', green_area)
        print('features in the blue area:', blue_area)


        if borutapy_green_blue == "both":
            borutapy_top_features = green_area + blue_area
        else:
            borutapy_top_features = green_area
            
        borutapy_top_features_df = pd.DataFrame(borutapy_top_features,columns = ['Feature'])
        borutapy_top_features_df['Method'] = 'Borutapy'
        
        return borutapy_top_features_df,borutapy

    #8. Select features based on BorutaShap method

    def borutashap_feature_selection(self, data, train_target,borutashap_estimator,borutashap_trials,borutashap_green_blue):
        
        #Inputs
        # data - Input feature data 
        # train_target - Target variable training data
        # borutashap_estimator - base model (default: XG Boost)
        # borutashap_trials -  number of iteration
        # borutashap_green_blue - choice for green and blue features
        

        ## Initialize borutashap
        
        if borutashap_estimator == "RandomForest":
            # Manual Change in Parameters - RandomForest
            estimator_borutashap = RandomForestClassifier(n_jobs = -1, random_state=101,max_depth=7, eval_metric='mlogloss')
        elif borutashap_estimator == "LightGBM":
            # Manual Change in Parameters - LightGBM
            estimator_borutashap = lgb.LGBMClassifier(n_jobs = -1, random_state=101,max_depth=7, eval_metric='mlogloss')
        else:
            # Manual Change in Parameters - XGBoost      
            estimator_borutashap = XGBClassifier(n_jobs = -1, random_state=101,max_depth=7, eval_metric='mlogloss')

        ## fit BorutaShap
        # Manual Change in Parameters - BorutaShap
        borutashap = BorutaShap(model = estimator_borutashap, importance_measure = 'shap', classification = True)
        borutashap.fit(X = data, y = train_target, n_trials = borutashap_trials)
        
        ## print results
        borutashap.plot(which_features = 'all')

        ## print results
        green_area = borutashap.accepted
        blue_area = borutashap.tentative
        print('features in the green area:', green_area)
        print('features in the blue area:', blue_area)


        if borutashap_green_blue == "both":
            borutashap_top_features = green_area + blue_area
        else:
            borutashap_top_features = green_area
            
        borutashap_top_features_df = pd.DataFrame(borutashap_top_features,columns = ['Feature'])
        borutashap_top_features_df['Method'] = 'Borutashap'

        
        return borutashap_top_features_df,borutashap
    
    #9. Select features based on Forward selection method

    def forward_selection(self, data, target, significance_level=0.05):

        #Inputs
        # data - Input feature data 
        # target - Target variable training data
        # significance_level -(e.g. SL = 0.05 with a 95% confidence)


        initial_features = data.columns.tolist()
        best_features = []
        while (len(initial_features)>0):
            remaining_features = list(set(initial_features)-set(best_features))
            new_pval = pd.Series(index=remaining_features)
            for new_column in remaining_features:
                model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
                new_pval[new_column] = model.pvalues[new_column]
            min_p_value = new_pval.min()
            if(min_p_value<significance_level):
                best_features.append(new_pval.idxmin())
            else:
                break
        return best_features

    #10. Select features based on backward elimination method

    def backward_elimination(self,data, target,significance_level = 0.05):

        #Inputs
        # data - Input feature data 
        # target - Target variable training data
        # significance_level -(e.g. SL = 0.05 with a 95% confidence)

        features = data.columns.tolist()
        while(len(features)>0):
            features_with_constant = sm.add_constant(data[features])
            p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
            max_p_value = p_values.max()
            if(max_p_value >= significance_level):
                excluded_feature = p_values.idxmax()
                features.remove(excluded_feature)
            else:
                break 
        return features
    
    #11. Select features based on Bi-directional elimination method

    def stepwise_selection(self, data, target,SL_in=0.05,SL_out = 0.05):

        #Inputs
        # data - Input feature data 
        # target - Target variable training data
        # SL_in -Perform the next step of forward selection (newly added feature must have p-value < SL_in to enter)
        # SL_out - Perform all steps of backward elimination (any previously added feature with p-value>SL_out is ready to exit the model)

        initial_features = data.columns.tolist()
        best_features = []
        while (len(initial_features)>0):
            remaining_features = list(set(initial_features)-set(best_features))
            new_pval = pd.Series(index=remaining_features)
            for new_column in remaining_features:
                model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
                new_pval[new_column] = model.pvalues[new_column]
            min_p_value = new_pval.min()
            if(min_p_value<SL_in):
                best_features.append(new_pval.idxmin())
                while(len(best_features)>0):
                    best_features_with_constant = sm.add_constant(data[best_features])
                    p_values = sm.OLS(target, best_features_with_constant).fit().pvalues[1:]
                    max_p_value = p_values.max()
                    if(max_p_value >= SL_out):
                        excluded_feature = p_values.idxmax()
                        best_features.remove(excluded_feature)
                    else:
                        break 
            else:
                break
        return best_features


