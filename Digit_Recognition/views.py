from joblib import load
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os

def load_model(model):
    return load(model)

model_save_dir = './exported_models'
def getHome(request):
    return render(request,'main.html')

def getBestPrediction(request):
    if request.method == "POST":
        algorithm = request.POST.get('algorithm')
        images = request.FILES.getlist('image')
        predictions = []
        for image in images:
            fs = FileSystemStorage(location='media/images')
            filename = fs.save(image.name, image)
            file_url = fs.url(filename)

            if algorithm == 'All':
                for saved_model in os.listdir(model_save_dir):
                    model = load_model(saved_model)
                    res = model.predict(data)
                    model_name = saved_model.split('.')[0]
                    predictions.append({'model': model_name, 'prediction': res})
            else:
                model = load_model(os.path.join(model_save_dir, 'knn_final.joblib'))
                res = model.predict(data)
                model_name = ("KNN-CV")
                predictions.append({'model': model_name, 'prediction': res})

        return render(request, 'main.html', {'predictions': predictions})
    else:
        return render(request, 'main.html')



def getAllPredictions(request):
    pass

def about(request):
    return render(request,'about.html')

def analytics(request):
    return render(request,'analytics.html')