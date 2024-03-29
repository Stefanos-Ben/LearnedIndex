from LearnedIndex import LearnedIndex, Regression
from DataGen import DataGen, Distribution
from helpers import hist_plot


def main():
    data = DataGen(Distribution.RANDOM, 1000000).generate()
    idx = LearnedIndex(Regression.RFR, data)  # Create a learned index for the samples created
    # res, err = idx.find(10)  # Find the position of the key using the index
    error_board = idx.find_all()
    error_board = error_board.astype(int)
    hist_plot(error_board, 'Random Forest Regression in Random Data N=1.000.000')


if __name__ == '__main__':
    main()
