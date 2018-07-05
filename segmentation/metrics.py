def intersection_over_union(y_true, y_score):
    y_true, y_score = y_true != 0, y_score != 0
    return (y_true & y_score).sum() / (y_true | y_score).sum()
