def prediction_model(pclass, sex, age, sibsp, parch, fare, embarked, title):
    import pickle
    x = [[pclass, sex, age, sibsp, parch, fare, embarked, title]]
    random_forest = pickle.load(open('titanic_model.sav', 'rb'))
    prediction = random_forest.predict(x)
    return prediction