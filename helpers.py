import matplotlib.pyplot as plt


def minmax(min_val, max_val, val):
    """
    Returns the value that falls within the range defined by min_val and max_val.

    This function ensures that the provided value 'val' lies within the range
    defined by 'min_val' and 'max_val'. If 'val' is less than 'min_val', it
    returns 'min_val'; if 'val' is greater than 'max_val', it returns 'max_val';
    otherwise, it returns 'val'.

    Args:
        min_val (float or int): The minimum value of the range.
        max_val (float or int): The maximum value of the range.
        val (float or int): The value to be checked and adjusted if necessary.

    Returns:
        float or int: The adjusted value that falls within the range [min_val, max_val].
    """
    if max(min_val, val) == val and min(val, max_val) == val:
        return val
    elif max(min_val, val) != val and min(val, max_val) == val:
        return min_val
    else:
        return max_val


def list_cut(ls):
    """
    Trims trailing zeros from a list.

    This function takes a list as input and removes any trailing zeros from the end
    of the list. It returns the modified list without the trailing zeros.

    Args:
        ls (list): The input list containing elements to be trimmed.

    Returns:
        list: The modified list with trailing zeros removed.
    """
    rls = list(reversed(ls))
    index = next((i for i, x in enumerate(rls) if x != 0), None)
    if index is None:
        return ls
    else:
        return list(reversed(rls[index:]))


def hist_plot(err_board, title):
    """
        Plots a histogram of errors occurrences.

        This function takes an error board containing error occurrences and plots
        a histogram representing the distribution of error counts. Each bin of the
        histogram represents the number of occurrences of errors.

        Args:
            err_board (list): A list containing error occurrences.
            title (str): The title of the histogram plot.

        Returns:
            None
        """
    errors = list_cut(err_board)
    occurrences = {}
    for i, count in enumerate(errors):
        occurrences[i] = count
    checks = list(occurrences.keys())
    counts = list(occurrences.values())
    plt.bar(checks, counts, align='center', color='red', edgecolor='black')
    plt.xlabel('Number of checks')
    plt.ylabel('Occurrences')
    plt.title(title)
    plt.show()
    print(errors)


def scatter_plot(x, y, y_pred, title):
    """
    Plots a scatter plot with a regression line.

    This function takes input features 'x', true target values 'y', predicted
    target values 'y_pred', and a title, and plots a scatter plot of the
    true values along with a regression line representing the predicted values.

    Args:
        x (array-like): The input features.
        y (array-like): The true target values.
        y_pred (array-like): The predicted target values.
        title (str): The title of the scatter plot.

    Returns:
        None
    """

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.scatter(x, y)
    plt.plot(x, y_pred, c="red")
    plt.show()
