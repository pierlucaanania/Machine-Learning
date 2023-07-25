import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

'''Creating Dataset'''
# Generate 1D Regression dataset
X, Y = make_regression(n_samples=2500,
                       n_features=1,
                       noise=20,
                       random_state=0)
# Plot the generated datasets
plt.scatter(X,Y)
plt.title('Regression Dataset')
plt.show()

'''Loss Function'''
def loss_function(m,b,points):
    total_error = 0
    for i in range(len(points)):
        x = X[i]
        y = Y[i]
        total_error += (y - (m*x+b))**2
    print(sum(total_error /  float(len(points))).round(2))

'''Gradient Descent'''

def gradient_descent(m_now, b_now, points, eta):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = X[i]
        y = Y[i]

        m_gradient += (-2/n)* x * (y-(m_now * x + b_now))
        b_gradient += (-2/n)* (y- (m_now*x + b_now))

    m = m_now - m_gradient * eta
    b = b_now - b_gradient * eta

    return m,b

'''Execute'''
m = 0
b = 0
eta = 0.001
epochs = 1100

for i in range(epochs):
    if i%50 == 0:
        print(f'Epoch: {i}')
    m,b = gradient_descent(m,b,X,eta)

print(m,b)


plt.scatter(X,Y)
plt.plot(list(range(-4,4)), [m*x + b for x in range(-4,4)], color = 'red')
plt.title('Linear Regression from Scratch')
plt.savefig('LinearRegression_Example')
plt.show()

