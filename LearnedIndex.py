import numpy as np
from helpers import minmax
from sklearn.linear_model import LinearRegression

class LearnedIndex:
    def __init__(self, data):
        self.data = np.array(data)
        self.data = np.sort(self.data)
        print(self.data)
        self.labels = np.where(~np.isnan(data))[0]  #Create
        self.keys = self.data[self.labels]
        self.build()

    def build(self):
        """
         Build an index for data and train a linear regression model.

         This function creates an index that maps labels to data, and then trains a linear regression model
         using the data and labels provided during object initialization.

         The resulting index is stored in the 'index' attribute, and the trained linear regression model
         is stored in the 'model' attribute.

         Note: Ensure that 'data' and 'labels' are already initialized before calling this function.

         Returns:
             None
         """
        self.index= {}
        for KEY, POS in zip(self.keys, self.labels):
            self.index[POS] = KEY
        X = self.keys.reshape(-1,1)
        Y = self.labels.reshape(-1,1)
        self.model = LinearRegression()
        self.model.fit(X,Y)


    def find(self,key):
        """
           Search for a key in an indexed data structure using a predictive model.

           Parameters:
               key: The value to search for in the data structure.

           Returns:
               (position, error): A tuple containing the position of the key in the data structure and the number of checks performed.
               If the key is not found after max_checks, the function returns (NaN, NaN).
        """
        upper_bound = len(self.index) - 1
        lower_bound = 0
        error = 0
        pos = minmax(lower_bound,upper_bound,np.rint(self.model.predict([[key]])[0][0]))
        # Holds the initial relativity of the key predicted with the key serached.
        esc_condition = self.index[pos] > key
        print(f"Model predicted that the requested key is in position {pos}")
        max_checks = 3

        while self.index[pos] != key:
            # Escape if you overextend in key.
            pos += 1 if self.index[pos] < key else -1
            if pos > upper_bound :
                print("heyo")
                esc = True
            elif  (~esc_condition and self.index[pos] <= key) or (esc_condition and self.index[pos] >= key):
                esc = False
                error += 1
            else:
                esc =True
                error += 1
            if ~(0 <= pos < len(self.data)) or esc:
                print(f"After making {error + 1} checks I figured that the key doesn't exist!")
                return np.NaN, np.NaN
        print(f"Found {key} in position {pos} after making {error + 1} checks")
        return pos, error