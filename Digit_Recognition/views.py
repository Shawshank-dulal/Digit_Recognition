from joblib import load
from django.shortcuts import render

def load_model(model):
    load(model)

def getHome(request):
    return render(request,'main.html')

def getBestPrediction(request):
    # return render(request,'main.html')
    if request.method == "POST":
        algorithm = request.POST.get('algorithm')
        image = request.FILES.get('image')
        print(algorithm)
        prediction = load_model(algorithm).predict(image) # make necessary change in this view
        return render(request,'main.html',{'prediction':prediction})
    else:
        return render(request,'main.html')


def getAllPredictions(request):
    pass

def about(request):
    return render(request,'about.html')

def analytics(request):
    return render(request,'analytics.html')