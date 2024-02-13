import matplotlib.pyplot as plt


def minmax(min_val, max_val, val):
    if max(min_val, val) == val and min(val, max_val) == val:
        return val
    elif max(min_val, val) != val and min(val, max_val) == val:
        return min_val
    else:
        return max_val


def hist_plot(err_board, title):
    occurrences = {}
    for i, count in enumerate(list_cut(err_board)):
        occurrences[i] = count
    checks = list(occurrences.keys())
    counts = list(occurrences.values())
    plt.bar(checks, counts, align='center', color='red', edgecolor='black')
    plt.xlabel('Number of checks')
    plt.ylabel('Occurrences')
    plt.title(title)
    plt.show()


def scatter_plot(x, y, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.scatter(x, y)
    plt.plot(x, y_pred, c="red")
    plt.show()


def list_cut(ls):
    rls = list(reversed(ls))
    index = next((i for i, x in enumerate(rls) if x != 0), None)
    if index is None:
        return ls
    else:
        return list(reversed(rls[index:]))

