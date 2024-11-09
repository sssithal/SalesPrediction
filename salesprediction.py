import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


advertising = pd.read_csv("advertising.csv")
# displays few lines of data set
advertising.head()

# create 3 sub-scatterplots
# note: tv has strong positive correlation, radio has somewhat, newspaper is random
sns.pairplot(
    advertising,
    x_vars=["TV", "Radio", "Newspaper"],
    y_vars="Sales",
    height=4,
    kind="scatter",
)
plt.show()

# creating correlation matrix to further confirm
# note: tv ~ 0.9, radio ~ 0.35, news ~ 0.16
sns.heatmap(advertising.corr(), cmap="YlOrRd", annot=True)
plt.show()

# create linear regression model
x = advertising["TV"]
y = advertising["Sales"]

# split variables into training and testing sets
# train = dataset to use for model
# test = data tested with model
# x = feature matrix, y = target variable

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, test_size=0.3, random_state=100
)

"""
print(x_train.shape) # print (number of samples, number of features)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
"""

# finding line of best fit
x_train_sm = sm.add_constant(
    x_train
)  # manually add constant attribute to find the intercept
lr = sm.OLS(
    y_train, x_train_sm
).fit()  # use ols (ordinary least squares) to generate line of best fit

lr.params  # print parameters (intercept and slope for TV)
print(lr.summary())  # print evaluation metrics for linear regression

"""
OLS Regression Results                            
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.816 - 81.6% of variance in sales is in tv
Model:                            OLS   Adj. R-squared:                  0.814
Method:                 Least Squares   F-statistic:                     611.2 - model fit is statistically significant, variance isnt  by chance.
                                       
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          6.9487      0.385     18.068      0.000       6.188       7.709
TV             0.0545      0.002     24.722      0.000       0.050       0.059
            low p value
          statistically significant

"""

# using sklearn to perform linear regression ---------------------------------------------


# predictions performed on test data
x_test_sm = sm.add_constant(x_test)  # add constant to x_test
y_pred = lr.predict(x_test_sm)  # predict y values corresponding to x_test_sm
#print(y_pred.head())  # predicted values

# check how well values are predicted on test data
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # mean squared error - how close regression line is to data points
print(r2_score(y_test, y_pred))  # R-squared value



x_train_lm = x_train.values.reshape(-1, 1)
x_test_lm = x_test.values.reshape(-1, 1)
y_train_lm = y_train.values.reshape(-1, 1)  # Reshape y_train

#print(x_train_lm.shape)
#print(x_train_lm.shape)

# representing LinearRegression as lr (creating LinearRegression object)
lr = LinearRegression()
# fit the model using lr.fit()
lr.fit( x_train_lm , y_train_lm)

#get intercept
print(lr.intercept_)
#get slope
print(lr.coef_)