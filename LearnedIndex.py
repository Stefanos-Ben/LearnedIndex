import numpy as np
from sklearn.linear_model import LinearRegression

class LearnedIndex:
    def __init__(self, data):
        self.data = data
        np.sort(self.data)
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
        error = 0
        pos = np.rint(self.model.predict([[key]])[0])
        max_checks = 3

        while self.index[pos[0]] != key:
            checks = 0
            error += 1
            pos += error if self.index[pos[0]] < key else -error
            checks += 1

            if checks >= max_checks:
                print("The key doesn't exist!")
                return np.NaN, np.NaN

        print(f"Found {key} in position {pos[0]} after making {error + 1} checks")
        return pos[0], error