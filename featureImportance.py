#Feature Importance
#Feature importance refers to a class of techniques for assigning scores to input features to a predictive model that indicates the relative #importance of each feature when making a prediction.

#Feature importance scores can be calculated for problems that involve predicting a numerical value, called regression, and those problems that #involve predicting a class label, called classification.

#Coefficients as Feature Importance
#Linear machine learning algorithms fit a model where the prediction is the weighted sum of the input values.

#Examples include linear regression, logistic regression, and extensions that add regularization, such as ridge regression and the elastic net.

#All of these algorithms find a set of coefficients to use in the weighted sum in order to make a prediction. These coefficients can be used #directly as a crude type of feature importance score.

#Let’s take a closer look at using coefficients as feature importance for classification and regression. We will fit a model on the dataset to #find the coefficients, then summarize the importance scores for each input feature and finally create a bar chart to get an idea of the #relative importance of the features.#




# linear regression feature importance

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBRegressor
from xgboost import XGBClassifier

from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

from matplotlib import pyplot

import statsmodels.formula.api as smf
import warnings

#suppress warnings
warnings.filterwarnings('ignore')


class FeatureImportance:

    def __init__(self):
        print("calling __init__() constructor...")

    #Linear Regression Feature Importance
    #We can fit a LinearRegression model on the regression dataset and retrieve the coeff_ property that contains 
    #the coefficients found for each #input variable.
    #These coefficients can provide the basis for a crude feature importance score. This assumes that 
    #the input variables have the same scale or #have been scaled prior to fitting a model.

    def LinearRegresssionFeatureImportance(self,X, y):
        #Inputs
        # X - Input feature data
        # y - Target Variable
        
        # define the model
        model = LinearRegression()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.coef_
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()

    #Logistic Regression Feature Importance
    #We can fit a LogisticRegression model on the regression dataset and retrieve 
    #the coeff_ property that contains the coefficients found for each input variable.
    #These coefficients can provide the basis for a crude feature importance score. 
    #This assumes that the input variables have the same scale or have been scaled prior to fitting a model.

    def logisticRegressionFeatureImportance(self,X, y):
        #Inputs
        # X - Input feature data
        # y - Target Variable

        # define the model
        model = LogisticRegression()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.coef_[0]
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()
    
    #CART Regression Feature Importance
    #The complete example of fitting a DecisionTreeRegressor and summarizing 
    #the calculated feature importance scores.

    def cartRegressionFeatureImportance(self,X, y):
        #Inputs
        # X - Input feature data
        # y - Target Variable

        # decision tree for feature importance on a regression problem
        # define the model
        model = DecisionTreeRegressor()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()
    
    #CART Classification Feature Importance
    #The complete example of fitting a DecisionTreeClassifier and 
    #summarizing the calculated feature importance scores.

    def cartClassificationFeatureImportance(self,X, y):

        #Inputs
        # X - Input feature data
        # y - Target Variable

        # define the model
        model = DecisionTreeClassifier()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()

    #Random Forest Regression Feature Importance
    #The complete example of fitting a RandomForestRegressor 
    #and summarizing the calculated feature importance scores.

    def randomForestRegressionFeatureImportance(self,X, y):

        #Inputs
        # X - Input feature data
        # y - Target Variable

        # define the model
        model = RandomForestRegressor()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()
    

    
    #Random Forest Classification Feature Importance
    #The complete example of fitting a RandomForestClassifier 
    #and summarizing the calculated feature importance scores.

    def randomForestClassificationFeatureImportance(self,X, y):

        #Inputs
        # X - Input feature data
        # y - Target Variable

        # define the model
        model = RandomForestClassifier()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()

    #XGBoost Regression Feature Importance
    #The complete example of fitting a XGBRegressor 
    #and summarizing the calculated feature importance scores.

    def xgBoostRegressionFeatureImportance(self,X, y):

        #Inputs
        # X - Input feature data
        # y - Target Variable

        # define the model
        model = XGBRegressor()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()

    
    #XGBoost Classification Feature Importance
    #The complete example of fitting an XGBClassifier 
    #and summarizing the calculated feature importance scores is listed below.

    def xgBoostClassificationFeatureImportance(self,X, y):

        #Inputs
        # X - Input feature data
        # y - Target Variable


        # define the model
        model = XGBClassifier()
        # fit the model
        model.fit(X, y)
        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()

    #Permutation Feature Importance for Regression
    #The complete example of fitting a KNeighborsRegressor 
    #and summarizing the calculated permutation feature importance scores is listed below.

    def permutationFeatureImportanceRegression(self,X, y):

        #Inputs
        # X - Input feature data
        # y - Target Variable


        # define the model
        model = KNeighborsRegressor()
        # fit the model
        model.fit(X, y)
        # perform permutation importance
        results = permutation_importance(model, X, y, scoring='neg_mean_squared_error')
        # get importance
        importance = results.importances_mean
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()

    #Permutation Feature Importance for Classification
    #The complete example of fitting a KNeighborsClassifier 
    #and summarizing the calculated permutation feature importance scores is listed below.

    def permutationFeatureImportanceClassification(self,X, y):

        #Inputs
        # X - Input feature data
        # y - Target Variable

        # define the model
        model = KNeighborsClassifier()
        # fit the model
        model.fit(X, y)
        # perform permutation importance
        results = permutation_importance(model, X, y, scoring='accuracy')
        # get importance
        importance = results.importances_mean
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()

    
    #Feature Selection with Importance
    #Feature importance scores can be used to help interpret the data, 
    #but they can also be used directly to help rank and select features that are most useful to a predictive model.



    # feature selection
    def select_features(self, X_train, y_train, X_test):

        #Inputs
        # X_train - Train input feature data
        # X_test - Test input feature data
        # y_train - Train Target Variable


        # configure to select a subset of features
        fs = SelectFromModel(RandomForestClassifier(n_estimators=1000), max_features=5)
        # learn relationship from training data
        fs.fit(X_train, y_train)
        # transform train input data
        X_train_fs = fs.transform(X_train)
        # transform test input data
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs

    def featureSelectionwithImportance(self,X, y):

        #Inputs
        # X - Input feature data
        # y - Target Variable

        # define the dataset
        #X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
        # feature selection
        X_train_fs, X_test_fs, fs = self.select_features(X_train, y_train, X_test)
        # fit the model
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train_fs, y_train)
        # evaluate the model
        yhat = model.predict(X_test_fs)
        # evaluate predictions
        accuracy = accuracy_score(y_test, yhat)
        return (accuracy*100)
        
        #print('Accuracy: %.2f' % (accuracy*100))
    def oddsRatios(self,data,varInput):

        #Inputs
        # data - Input feature dataframe
        # varInput - columns to columns to parse

        log_reg = smf.logit( varInput, data=data).fit()
        # ... Define and fit model
        odds_ratios = pd.DataFrame(
            {
                "OR": log_reg.params,
                "Lower CI": log_reg.conf_int()[0],
                "Upper CI": log_reg.conf_int()[1],
            }
        )
        #odds_ratios = np.exp(odds_ratios)
        return odds_ratios

                    




