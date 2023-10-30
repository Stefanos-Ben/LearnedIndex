def minmax(min_val,max_val,val):
    if max(min_val, val) == val and min(val, max_val) == val:
        return val
    elif max(min_val, val) != val and min(val, max_val) == val:
        return min_val
    else:
        return max_val

