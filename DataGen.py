import numpy as np
from enum import Enum


class Distribution(Enum):
    LINEAR = 0
    NORMAL = 1
    RANDOM = 2


class DataGen:
    def __init__(self, distribution, size):
        self.data = None
        self.distribution = distribution
        self.size = size

    def generate(self):
        """
        Generates data based on the specified distribution.

        This method generates data points based on the distribution specified in the
        'distribution' attribute of the object. The supported distributions are:
        - LINEAR: Generates linear data points.
        - NORMAL: Generates data points following a log-normal distribution.
        - RANDOM: Generates random data points.

        Returns:
            data: An array containing the generated data points.

        Note:
            Ensure that the 'distribution' attribute is set to one of the supported
            distributions before calling this method. If an unsupported distribution
            is provided, the method will print an error message and return None.
        """
        if self.distribution == Distribution.LINEAR:
            self.data = self.linear_data()
        elif self.distribution == Distribution.NORMAL:
            self.data = self.log_normal()
        elif self.distribution == Distribution.RANDOM:
            self.data = self.rand_data()
        else:
            print("I can't recognize the given distribution please give one of those(linear, normal, random)")
            return None
        return self.data

    def linear_data(self):
        """
        Generates linear data points based on a randomly generated linear equation.

        This method creates a set of data points that lie along a straight line.
        The equation of the line is randomly generated in the form 'y = ax + b',
        where 'a' and 'b' are randomly chosen integers.

        Returns:
            data: An array containing data points generated along the linear line.
        """
        a = np.random.randint(10)
        b = np.random.randint(10)
        print(f"Creating linear data according to the {a}x+{b} line.")
        data = []
        for i in range(self.size):
            data.append(a*i+b)
        return np.array(data)

    def log_normal(self):
        """
           Generates an array of log-normal distributed values and scales them.

           Parameters:
           amount (int): The number of log-normal samples to generate.

           Returns:
           numpy.ndarray: An array of log-normal samples, scaled by the minimum value.
        """
        data = []
        for i in range(self.size):
            sample = 10. + np.random.standard_normal(100)
            data.append(np.prod(sample))
        data = np.array(data) / np.min(data)
        return np.rint(data)

    def rand_data(self):
        """
            Generates an array of randomly distributed values.

            Returns:
            numpy.ndarray: An array of random samples.
        """
        return np.random.randint(1, self.size*1000, self.size)
