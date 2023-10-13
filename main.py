import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def logNormalGen(amount):
    data = []
    for i in range(amount):
        sample = 10. + np.random.standard_normal(100)
        data.append(np.prod(sample))
    data = np.array(data)/np.min(data)
    return data


def lineDataGen(amount, a, b):
    data=[]
    data.append(0)
    for i in range(amount-1):
        data.append(a*i+b)
    return np.array(data)


def train(data, positions):
    X = np.array(data).reshape(-1,1)
    Y = np.array(positions).reshape(-1,1)
    reg = LinearRegression()
    reg.fit(Y,X)
    plt.scatter(Y,X)
    plt.show()
    return reg

def main():

    linear = lineDataGen(100, 2, 1)
    labels = np.arange(len(linear), dtype=np.int32)
    linearDf = pd.DataFrame(linear)
    model = train(linear, labels)
    predictions = model.predict(linear.reshape(-1,1))
    # predictions = predictionRound(predictions)
    print(linear)
    print(np.rint(predictions))
    print(len(predictions))
    print(len(linear))
    plt.scatter(labels.reshape(-1,1), linear.reshape(-1,1))
    plt.scatter(labels.reshape(-1,1), predictions, color='red')
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