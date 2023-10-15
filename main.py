import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def logNormalGen(amount):
    """
       Generates an array of log-normal distributed values and scales them.

       Parameters:
       amount (int): The number of log-normal samples to generate.

       Returns:
       numpy.ndarray: An array of log-normal samples, scaled by the minimum value.
    """
    data = []
    for i in range(amount):
        sample = 10. + np.random.standard_normal(100)
        data.append(np.prod(sample))
    data = np.array(data)/np.min(data)
    return data


def lineDataGen(amount, a, b):
    """
    Generates linear data and labels.

    Parameters:
    amount (int): The number of data points to generate.
    a (float): The slope of the linear relationship.
    b (float): The y-intercept of the linear relationship.

    Returns:
    numpy.ndarray, numpy.ndarray: Arrays containing generated data and labels.
    """
    data= []
    labels = []
    for i in range(amount):
        data.append(i)
        labels.append(a*i+b)
    return np.array(data), np.array(labels)


def train(data, positions):
    """
    Trains a linear regression model and visualizes the data.

    Parameters:
    data (list or array-like): Input data.
    positions (list or array-like): Corresponding positions or labels for the data.

    Returns:
    sklearn.linear_model.LinearRegression: Trained linear regression model.
    """
    X = np.array(data).reshape(-1,1)
    Y = np.array(positions).reshape(-1,1)
    reg = LinearRegression()
    reg.fit(X,Y)
    plt.scatter(X,Y)
    plt.title("Data")
    plt.show()
    return reg


def modelPrecision(predictions, labels):
    score = 0
    for i in range(100):
        if predictions[i][0] == labels[i][0]:
            score = score + 1
    return score

def main():

    keys,labels = lineDataGen(100, -5, 3) #Generate 100 samples on the 2x + 1 line
    model = train(keys, labels) # Train a linear Regression Model on the generated keys and labels
    predictions = model.predict(keys.reshape(-1,1)) # Predict the position of the keys generated with the model.
    print(keys)
    print(labels)
    print(predictions)
    score = modelPrecision(np.rint(predictions), labels.reshape(-1,1)) # Calculate precision after rounding the samples
    print(str(score) + ' out of ' + str(len(keys)) + ' rounded samples predicted right')
    plt.scatter(keys.reshape(-1,1), predictions, color='red') # Plot the predictions.
    plt.title("Predictions")
    plt.show()

if __name__ == '__main__':
    main()


    # logNormal = logNormalGen(1000)
    # count, bins, ignored = plt.hist(logNormal, 100, density=True, color='green')

    # sigma = np.std(np.log(logNormal))
    # mu = np.mean(np.log(logNormal))
    #
    # x = np.linspace(min(bins), max(bins), 10000)
    # pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
    #    / (x * sigma * np.sqrt(2 * np.pi)))
    #
    # plt.plot(x, pdf,color='black')
    # plt.grid()
    # plt.show()

    # logNormal.sort()
    # logNormalDf = pd.DataFrame(logNormal)
    # print(logNormalDf)
    #
    # labels = np.arange(len(logNormal), dtype=np.float32)
    #
    # model = train(logNormal, labels)
    # print(model.predict([3.,1.]))