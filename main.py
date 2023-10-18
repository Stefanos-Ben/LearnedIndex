import numpy as np
import matplotlib.pyplot as plt
from LearnedIndex import LearnedIndex

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





def main():
    keys,labels = lineDataGen(10, 2, 1) #Generate 100 samples on the 2x + 1 line
    idx = LearnedIndex(keys, labels) # Create a learned index for the samples created
    res, err = idx.find(6) # Find the position of the key using the index
    print(res)


if __name__ == '__main__':
    main()



    # The size should be big enough so that the positions are enough according to the line.
    #index_size = 2 * (self.data[len(self.data) - 1]) + 1





