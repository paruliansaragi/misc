from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
import random 

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype = np.float64)

def create_dataset(hm, variance, step=2, correlation=False):
	val = 1
	ys = []
	for i in range(hm):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val+=step
		elif correlation and correlation == 'neg':
			val -= step

	xs = [i for i in range(len(ys))]

	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)



# y = mx + b
# to calculate m and b
# m = slope 
# m = mean of all x's * y's minus the mean of all the x's times the y's / mean of x's squared - mean of x squared
# b = y intercept
# b = mean of y's minus m times means of x's 

def best_fit_slope_and_intercept(xs,ys):
	m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) / 
		   ((mean(xs) * mean(xs)) - mean(xs*xs)))
	b = mean(ys) - m*(mean(xs))
	return m, b ##Algorithim for M = slope and b = y intercept



#r2 or coefficient of determination
#r2 is the squared error
#error is the distance from line of best fit 
#error distance may be negative or positive so we square it
#outlier : linear data set shouldn't have an outlier
#we square the error so we want to penalise for outliers
#you can not just square but 4, 6, 18 to penalise more for the outlier
#r2 = 1 - SE Yhat / SE mean of Ys in dataset 
#SE = Squared Error

def squared_error(ys_orig, ys_line):
	return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
	y_mean_line = [mean(ys_orig) for y in ys_orig]#for every Y we have find mean value of y
	squared_error_regr = squared_error(ys_orig, ys_line)
	squared_error_y_mean = squared_error(ys_orig, y_mean_line)
	return 1 - (squared_error_regr / squared_error_y_mean)

xs, ys = create_dataset(40, 40, 2, correlation='pos')



m, b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m*x)+b for x in xs]

predict_x = 8
predict_y = (m*predict_x)+b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, s=100, color='g')
plt.plot(xs, regression_line)
plt.show()




