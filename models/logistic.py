from sklearn.linear_model import LogisticRegression

def get_model():
    return LogisticRegression(max_iter=500, solver='liblinear')
