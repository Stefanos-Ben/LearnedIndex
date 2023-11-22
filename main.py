import numpy as np
import matplotlib.pyplot as plt
from LearnedIndex import LearnedIndex
from DataGen import DataGen, Distribution


def main():
    data = DataGen(Distribution.RANDOM, 100).generate()
    print(data)
    idx = LearnedIndex(data)  # Create a learned index for the samples created
    res, err = idx.find(10)  # Find the position of the key using the index


if __name__ == '__main__':
    main()







