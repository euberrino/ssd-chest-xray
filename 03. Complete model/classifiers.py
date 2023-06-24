from keras.models import load_model
import tensorflow as tf
from numpy import array
import torch


def load_img(fetch_dir):
    img = tf.keras.preprocessing.image.load_img(fetch_dir,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)/255
    input_arr = array([input_arr]) 
    return input_arr
    
def bietapic_1(fetch_dir): 
    img = load_img(fetch_dir)
    bietapic_1 = load_model('Filtro_Bietapico_Primero.h5')
    y = bietapic_1.predict(img,workers=8)
    biet_1_dict = {0: 'Chest', 1: 'Other'}
    preds = y.argmax(axis=-1)
    preds_c = [biet_1_dict[p] for p in preds]
    return y[0][0],preds_c[0]

def bietapic_2(fetch_dir):
    img = load_img(fetch_dir)
    bietapic_1 = load_model('Filtro_Bietapico_Segundo.h5')
    y = bietapic_1.predict(img,workers=8)
    biet_1_dict = {0: 'AP_horizontal', 1: 'L', 2: 'PA'}
    preds = y.argmax(axis=-1)
    preds_c = [biet_1_dict[p] for p in preds]
    return y[0],preds_c[0]

def opacity_detector(fetch_dir):
    img = load_img(fetch_dir)
    img = tf.keras.preprocessing.image.array_to_img(img[0])
    model = torch.hub.load('ultralytics/yolov5','custom',path='corrida_3.pt')
    results = model(img)
    print(results.pandas().xyxy[0])
    return results.pandas().xyxy[0]

