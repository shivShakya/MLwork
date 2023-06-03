from flask import Flask,jsonify,request,send_file
import cv2
import pickle
import numpy as np
from flask_cors import CORS
from PIL import Image


app = Flask("__main__") 
CORS(app)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
     
     file = request.files['image']
     file.save("imagefile")
     img = Image.open(file)
     img.save('predicted.jpg')
     
     Img_size = 100
     image=cv2.imread("/Users/jyoti-alok/Desktop/ML class/Animals api/predicted.jpg")
     new_array=cv2.resize(image,(Img_size,Img_size))
     new_array = new_array.reshape((1, 30000))
     y_pred = model.predict(new_array)
     print(y_pred)
     if (y_pred == [0]):
              y = "Dog"
     elif (y_pred == [1]):
              y = "Cat"
     else:
             y = "Panda"
    
     return y



if __name__ == "__main__":
       print("flask Api is running on the server port 5000")
       app.run()