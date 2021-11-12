from django.shortcuts import render
from . import fake_model
from . import ml_predict

def home(request):
    return render(request, 'index.html')

def result(request):


    pclass = int(request.GET["pclass"])
    user_sex = int(request.GET["sex"])
    user_age = int(request.GET["age"])
    sibsp = int(request.GET["sibsp"])
    parch = int(request.GET["parch"])
    fare = int(request.GET["fare"])
    embarked = int(request.GET["embarked"])
    title = int(request.GET["title"])
    

    prediction = ml_predict.prediction_model( pclass, user_sex, user_age, sibsp, parch, fare, embarked, title)
    if prediction ==1:
        survival = "You survived!"
    else:
        survival = "You didn't make it :("
    return render(request, 'result.html', { 'survival':survival, 'prediction':prediction})
