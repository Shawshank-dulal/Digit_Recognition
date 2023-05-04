from joblib import load
from django.shortcuts import render

def load_model(model):
    load(model)


def getHome(request):
    return render(request,'main.html')

def getBestPrediction(request):
    return render(request,'main.html')

def getAllPredictions(request):
    pass

def about(request):
    return render(request,'about.html')

def analytics(request):
    return render(request,'analytics.html')