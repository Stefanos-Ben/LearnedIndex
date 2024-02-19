import numpy as np
from helpers import minmax, scatter_plot
from enum import Enum
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


class Regression(Enum):
    LINEAR = 0
    POLYNOMIAL = 1
    RFR = 2


class LearnedIndex:
    def __init__(self, reg, data):
        self.poly = None
        self.reg = reg
        self.model = None
        self.index = None
        print(data)
        self.keys = np.sort(data)
        self.labels = np.arange(len(data))
        self.X = self.keys.reshape(-1, 1)
        self.Y = self.labels.reshape(-1, 1)
        self.build()

    def build(self):
        """
            Builds a regression model based on the specified regression technique.

            This method builds a regression model based on the specified regression
            technique stored in the 'reg' attribute of the object. It supports three
            types of regression: Linear Regression, Polynomial Regression, and Random
            Forest Regression.

            Returns:
                None

            """
        self.index = {}
        for KEY, POS in zip(self.keys, self.labels):
            self.index[POS] = KEY
        match self.reg:
            case Regression.LINEAR:
                self.model = LinearRegression()
                self.model.fit(self.X, self.Y)
            case Regression.POLYNOMIAL:
                degree = self.poly_degree()
                self.poly = PolynomialFeatures(degree=degree, include_bias=False)
                X_poly = self.poly.fit_transform(self.X)
                self.model = LinearRegression()
                self.model.fit(X_poly, self.Y)
            case Regression.RFR:
                rf = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
                grid_search.fit(self.X, self.Y.ravel())
                print("Best Parameters for Random Forest:", grid_search.best_params_)
                self.model = grid_search.best_estimator_

    def find(self, key):
        """
           Search for a key in an indexed data structure using a predictive model.

           Parameters:
               key: The value to search for in the data structure.

           Returns:
               (position, error): A tuple containing the position of the key in the data structure and the number of
               checks performed.
               If the key is not found after max_checks, the function returns (NaN, NaN).
        """
        upper_bound = len(self.index) - 1
        lower_bound = 0
        error = 0
        match self.reg:
            case Regression.LINEAR:
                inp = np.array([[key]]).reshape(-1, 1)
                pos = minmax(lower_bound, upper_bound, np.rint(self.model.predict(inp)[0][0]))
            case Regression.POLYNOMIAL:
                inp = self.poly.transform(np.array([[key]]))
                pos = minmax(lower_bound, upper_bound, np.rint(self.model.predict(inp)[0][0]))
            case Regression.RFR:
                inp = np.array([[key]]).reshape(-1, 1)
                pos = minmax(lower_bound, upper_bound, np.rint(self.model.predict(inp)[0]))
            case _:
                print("Unrecognized Model. Can't perform find operation")
                return None

        pred_pos = pos
        # Holds the initial relativity of the key predicted with the key searched.
        esc_condition = self.index[pos] > key
        # print(f"Model predicted that the requested key is in position {pos}")
        while self.index[pos] != key:
            # Escape if you overextend in key.
            pos += 1 if self.index[pos] < key else -1
            if pos > upper_bound or pos < lower_bound:
                esc = True
            elif (~esc_condition and self.index[pos] <= key) or (esc_condition and self.index[pos] >= key):
                esc = False
                error += 1
            else:
                esc = True
                error += 1
            if esc:
                print(f"After making {error + 1} checks I figured that the key doesn't exist!")
                return np.NaN, np.NaN
        # print(f"Found {key} in position {pos} after making {error + 1} checks")
        return pred_pos, error

    def find_all(self):
        """
        Finds predictions and error statistics for all keys.

        This method iterates through all keys in the dataset, finding predictions
        and error statistics for each key using the 'find' method. It then generates
        a scatter plot to visualize the predictions against the true values and
        returns error statistics.

        Returns:
            numpy.ndarray: An array containing error statistics for each key.
        """
        predictions = []
        stats = np.zeros(len(self.keys))
        for i in range(len(self.keys)):
            pred, err = self.find(self.keys[i])
            stats[err] += 1
            predictions.append(pred)
        scatter_plot(self.keys.reshape(-1, 1), self.Y, predictions, "Random Forest Regression in Random Data N=1.000.000")
        return stats

    def poly_degree(self):
        """
        Determines the optimal polynomial degree for polynomial regression.

        This method iterates through polynomial degrees from 1 to 5, fits polynomial
        regression models with each degree, and evaluates their performance using
        cross-validation with R^2 scoring. It returns the degree that yields the
        highest average R^2 score.
        """
        best_degree = None
        best_score = float('-inf')
        for degree in range(1, 6):
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(self.X)
            model = LinearRegression()
            scores = cross_val_score(model, X_poly, self.Y, cv=5, scoring='r2')
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_degree = degree
        print(f"Best degree: {best_degree} with average R^2 score: {best_score}")
        return best_degree
