def test(model, X_test):
    print("\n> Testing the model")
    predictions = model.predict(X_test)
    print('\033[92m'+"Testing ended successfully\n"+'\033[0m')
    return predictions

if __name__ == '__main__':
    test()