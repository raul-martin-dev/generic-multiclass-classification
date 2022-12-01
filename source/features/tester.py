def test(model, X_test):
    predictions = model.predict(X_test)
    return predictions

if __name__ == '__main__':
    test()