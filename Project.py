import pandas as pd                                                             # library used for data science in Python
import matplotlib.pyplot as plt                                                 # vizualization tool for plotting data points


data = pd.read_csv('data.csv')                                                  # reading the data from the associated CSV file
                                                                                # could also create an array in the code but limits 
                                                                                # scalability of data

def gradient_descent(m_now, b_now, points, L):                                  # Function to perform gradient descent to find the optimal values of slope (m) and intercept (b)
    m_gradient = 0                                                              # "L" is learning rate                    
    b_gradient = 0                                                              # Initial gradient values
    
    n = len(points)                                                             # number of data points
    
    for i in range(n):                                                          #iterate over each data point to calculate gradients
        x = points.iloc[i].x                                                    # get value of x                       
        y = points.iloc[i].y                                                    # get value of y
        
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))                    # Partial derivative of loss function with respect to m
        m_gradient += -(2/n) * (y - (m_now * x + b_now))                        # Partial derivative of loss function with respect to b
        
    m = m_now - m_gradient * L                                                  # Update m and b using gradients and learning rate "L"
    b = b_now - b_gradient * L
    return m, b

m = 0                                                                           # Initialize slope (m), intercept (b), learning rate (L), and number of epochs
b = 0
L = 0.0001                                                                      # learning rate is set between 0.0 and 1.0, higher numbers result in faster learing
epochs = 300                                                                    # but may not arrive at an optimal number, our data set isn't that big, so time isn't an issue

for i in range(epochs):                                                         # perform gradient descent for a certain number of epochs
    if i % 50 == 0:                                                             # an "epoch" in machine learning means one complete pass of the training dataset through the algorithm
        print(f"Epoch: {i}")                                                    # The amount of epochs determines 
    m, b = gradient_descent(m, b, data, L)
    
print(m, b)                                                                     # print final values of m and b

plt.scatter(data.x, data.y, color="black")                                      # Plot the original data points and line of best fit
plt.plot(list(range(0, 100)), [m * x + b for x in range(0, 100)], color="red")  # Scatter plot of data points
plt.show()                                                                      # Display final result