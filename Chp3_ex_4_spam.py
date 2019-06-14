# Ignore useless warnings 
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import numpy as np
import matplotlib.pyplot as plt
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)

#Plots the line of best fit
#plt.plot(X_new, y_predict, "r-")
#plt.plot(X, y, "b.")
#plt.axis([0, 2, 0, 15])
#plt.show()
#can do the above with below
#line reg equat yhat = theta0 (bias) + theta1x1 + .. thetanxn
#vectorised linreg is yhat = htheta(x) = theta transpose the dot product of x
#How well did the model fit the data? How to set the params so that the model best fits the training set
#We use the Mean Square Error (MSE) 
#MSE cost function: MSE(X, htheta) = 1/m sum m i=1 (theta transpose dot product x(i) -y(i))2
#To find the value of theta that mins the cost function we can use the Normal equation: theta hat = (X transpose dot product X)-1 dot product X transpose dot product y
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
#print(lin_reg.intercept_, lin_reg.coef_)  ~~ [4.11253303] [[2.76190641]]
#print(lin_reg.predict(X_new))
#[[ 4.18677157]
# [10.08491783]]

#Some level of computational complexity with using the normal equation
#The normal equation gets very slow when the numb of features grows large
# ~~ Gradient Descent ~~
#Tweaking parameters iteratively to minimize the cost function
#Direction of steepest descent
#Error function is the line function and it measures the local gradient of that error function with regards
#to the parameter vector theta and goes in the direction of descending gradient until the gradient is 0
#You start by filling theta with random values i.e. random initialisation and take gradual steps until convergence
#The learning rate hyperparameter determines the size of the steps
#THe MSE is a convex cost function i.e. one global minimum
#Use feature scaling i.e. StandardScaler for GD

# ~~ Batch gradient descent ~~
#to implement GD you need to compute the gradient of the cost function with regards to each model param thetaj
# in other words you need to calculate how much the cost functin will change in response to a small change in thetaj
# THis is called the partial derivative i.e. "what is the slope of the mountain under my feet if i face east? What about West/North/etc"
#Equation: d/d thetaj MSE(theta) = 2/m sum m i=1(theta Transpose dot product x(i) - y(i) )x(i)j
#We compure the partial derivaties all in one go for each model parameter  ~~ Hence batch GD
#Once you have the gradient vector that points uphill go in the opposite direction downhill
#This means subtracting Vthetasubscript MSE(theta) from theta 
#Multiply the gradient vector by n (i.e learning rate) to determine the size of the downhill step

eta = 0.1 #learning rate
n_iterations = 1000 
m = 100 #training examples
theta = np.random.randn(2,1) #random initialised theta

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) #gradients is the partial derivative the direction of steepest descent
    theta = theta - eta * gradients #then theta that mins cost function is theta minus learning rate times the gradient
#eta * gradients determines the size of downhill step
#print(theta)
#Use GridSearch to find the learning rate
#~~ Stochastic Gradient Descent ~~
# Batch can be too time consuming
# Stochastic GD picks a random instance in the training set at every step and computes the gradients based on that one instance
# Good for large training sets only one instance needs to be in memory at each iteration
# due to its random (stochastic) nature it bounces around and once it gets close to the minimum it bounces out again
#so once it stops the final params are good but not optimal
#One solution is to gradually reduce the learning rate step size. This is called "simulated annealing"
#the function that determines the learning rate at each iteration is called the learning schedule
n_epochs = 50
t0, t1 = 5, 50 #learning schedule hyperparameters
def learning_schedule(t):
	return t0 / (t + t1)

theta = np.random.randn(2,1)#random init

for epoch in range(n_epochs):
	for i in range(m):
		random_index = np.random.randint(m)
		xi = X_b[random_index:random_index+1]
		yi = y[random_index:random_index+1]
		gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
		eta = learning_schedule(epoch * m + i)
		theta = theta - eta * gradients

#print(theta)
#To use SGD with lin reg use SGDRegressor 
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, random_state=42) #runs 50 epochs, learning rate 0.1, default learning schedule and no regularisation i.e. penalty=none
sgd_reg.fit(X, y.ravel())
#print(sgd_reg.intercept_, sgd_reg.coef_)

#~~ Mini Batch GD ~~
'''
Computes the gradients on small random sets of instances - so not just on one (SGD) or the whole thing (BGD).
less erratic than SGD and closer to the minimum. BUT, harder to escape from local minima when problems suffer from local minima.
Algorithm 	large (m) 	out-of-core support 	large (n)	hypereparams	scaling required	scikit-learn
~Normal equation Fast	No 						slow        	0 				no 					LinearRegression
~ Batch GD  	slow     	no 						fast        2               Yes                   n/a
SGD              fast        yes                   fast        >=2                yes                  SGDRegressor
Mini-batch GD     fast          yes                 fast        >=2               yes                   SGDRegressor
'''

''' ~~ Polynomial Regression ~~

What is data is nonlinear ~ well you can fit linear regression to non-linear data by:
adding the powers of each feature as new features => polynomial regression.
Lets start by creating some non-linear data
'''
import numpy as np
import numpy.random as rnd

np.random.seed(42)
m = 100 
X = 6 * np.random.randn(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
'''
Lets use scikit learns PolynomialFeatures class to transform our training data
 Polynomial with degree=3 adds not only features a2, a3, b2, b3 but also the combination ab, a2b, ab2
 PolynomialFeatures(degree=d) transforms an array containing n features into an array containing n! where
 n!= n factorial of n equal to 1 x 2 x 3 x ... x n => beware of the combinatorial feature explosion
'''
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
#print(X[0]) [-0.7781204]
#print(X_poly[0])

#Plotting the linear regression agains the poly
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
X_new=np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

#plt.plot(X, y, "b.")
#plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
#plt.xlabel("$x_1$", fontsize=18)
#plt.ylabel("$y$", rotation=0, fontsize=18)
#plt.legend(loc="upper left", fontsize=14)
#plt.axis([-3, 3, 0, 10])
#plt.show()

'''
Learning Curves
 High degree of poly will fit much better than lin reg 
 lin reg underfit ~ poly overfit => quadratic best fit
We used cross-validation to estimate a model's generalisation performance. If a model performs well on the training data but 
generalises poorly according to cross-valdiation then your model is overfitting, if it performs poorly on both then it is underfit.
Another way to measure this is learning curves. These are plots of the models performance on the training set and validation set as
a function of the training set size (or the training iteration). 
'''