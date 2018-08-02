import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression 

boston = load_boston()
boston.keys() #get the keys of this dictionary
bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names #replace numbers with feature names
price = boston.target #this contains the prices
bos['PRICE'] = price #add price to the bos dataset

X = bos.drop('PRICE', axis=1) #drop price and get only the independent parameters
lm = LinearRegression()
lm.fit(X, bos.PRICE)

lm.intercept_ #get the intercept 
lm.coef_ #get the coefficients - should be 13 in number (i.e. minus PRICE)

#put the features and coefficients in a dataFrame
pd.DataFrame(zip(X.columns, lm.coef_), columns=['features', 'coefficients']) 

#Plot RM vs Price because RM has the highest correlation coefficient
plt.scatter(bos.RM, bos.PRICE)
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("Housing Price")
plt.title("Relationship between RM and Price")
#plt.show()

lm.predict(X)[0:5]
#plot a scatter diagram of predicted vs prices
plt.scatter(bos.PRICE, lm.predict(X))
plt.xlabel('Prices')
plt.ylabel('Predicted prices')
plt.title('Prices vs Predicted Prices')
#plt.show()

mseFull = np.mean((bos.PRICE - lm.predict(X)) ** 2) #The error is big
#print mseFull

#split PRICE and predicted X into training and testing (not recommended)
#This is for just test. You ought to create this diff. sets manually
# X_train = X[:-50]
# X_test = X[-50:]
# Y_train = bos.PRICE[:-50]
# Y_test = bos.PRICE[-50:]

#The best way to split into test and train is use the scikit's train_test_split
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
	X, bos.PRICE, test_size=0.33, random_state = 5)

lm.fit(X_train, Y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)

#Residual Plot
plt.scatter(lm.predict(X_train), lm.predict(X_train) - Y_train, c='b', s=40, alpha=0.5)
plt.scatter(lm.predict(X_test), lm.predict(X_test) - Y_test, c='g', s=40)
plt.hlines(y = 0, xmin=0, xmax=50)
plt.title('Residual Plot using training (blue) and test(green) data')
plt.ylabel('Residuals')
#plt.show()









