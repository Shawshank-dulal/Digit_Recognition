from joblib import load
from PIL import Image
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os


def load_model(model):
    return load(model)

def process_data(file_url):
    image=Image.open(file_url)
    scaler=load('scaler.save')
    
    image_df=pd.DataFrame(np.array(image).reshape(1,784),columns=[f'pixel{str(i)}' for i in range(784)])
    scaler.fit_transform(image_df)
    return image_df

model_save_dir = 'exported_models'


def getHome(request):
    return render(request, 'main.html')


def getBestPrediction(request):
    if request.method == "POST":
        algorithm = request.POST.get('algorithm')
        images = request.FILES.getlist('image')
        print(algorithm)
        predictions = []
        for image in images:
            fs = FileSystemStorage(location='media/images')
            filename = fs.save(image.name, image)
            file_url = fs.url(filename)
            data=process_data(f'media/images{file_url}')
            if algorithm == 'all':
                for saved_model in os.listdir(model_save_dir):
                    model = load_model(os.path.join(model_save_dir,saved_model))
                    res = model.predict(data)
                    print(res)
                    model_name = saved_model.split('.')[0]
                    predictions.append({'image_dir':f'media/images{file_url}','model': model_name, 'prediction': res})
            else:
                model = load_model(os.path.join(model_save_dir,'KNN-CV.joblib'))
                res = model.predict(data)
                model_name = ("KNN-CV")
                
                predictions.append({'image_dir':f'media/images{file_url}','model': model_name, 'prediction': res})

        return render(request, 'main.html', {'predictions': predictions})
    else:
        return render(request, 'main.html')


def getAllPredictions(request):
    pass


def about(request):
    return render(request, 'about.html')


def analytics(request):
    return render(request, 'analytics.html')