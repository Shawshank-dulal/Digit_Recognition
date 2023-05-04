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
    dir_path = 'static/images/'
    for file_name in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, file_name))
    if request.method == "POST":
        algorithm = request.POST.get('algorithm')
        images = request.FILES.getlist('image')
        predictions = []
        for image in images:
            fs = FileSystemStorage(location='static/images')
            filename = fs.save(image.name, image)
            file_url = fs.url(filename)
            data=process_data(f'static/images{file_url}')
            if algorithm == 'all':
                for saved_model in os.listdir(model_save_dir):
                    model = load_model(os.path.join(model_save_dir,saved_model))
                    res = model.predict(data)[0]
                    model_name = saved_model.split('.')[0]
                    predictions.append({'image_dir':'images'+file_url,'model': model_name, 'prediction': res})
            else:
                model = load_model(os.path.join(model_save_dir,'KNN-CV.joblib'))
                res = model.predict(data)[0]
                model_name = ("KNN-CV")
                
                predictions.append({'image_dir':'images'+file_url,'model': model_name, 'prediction': res})
        print(predictions)
        return render(request, 'main.html', {'predictions': predictions})
    else:
        return render(request, 'main.html')




def about(request):
    return render(request, 'about.html')


def analytics(request):
    model_stats=[{'model':'Logistic Regression' , 'accuracy': 0.91, 'precision': 0.92, 'recall': 0.92 , 'f1': 0.92, 'image':'plots/logreg.png'}, 
                  {'model': 'Decision Tree' , 'accuracy': 0.85, 'precision': 0.86 , 'recall':0.86 , 'f1':0.86, 'image':'plots/dtree.png'}, 
                  {'model': 'KNN', 'accuracy':0.9, 'precision': 0.91, 'recall':0.92 , 'f1':0.91, 'image':'plots/knn.png'}, 
                  {'model': 'KNN CV', 'accuracy':0.92, 'precision': 0.93, 'recall':0.93, 'f1':0.93, 'image':'plots/knn-cv.png'}, 
                  {'model':'SVM', 'accuracy':0.97, 'precision':0.97 , 'recall': 0.97, 'f1':0.97, 'image':'plots/svm.png'}]
    return render(request, 'analytics.html', {'models_data':model_stats})