from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import io
import cv2
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from flask_cors import CORS,cross_origin



emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
#make shots directory to save pics
# try:
#     os.mkdir('./shots')
# except OSError as error:
#     pass

#Load pretrained face detection model    
facecasc = cv2.CascadeClassifier('./static/haarcascade_frontalface_default.xml')
#load pretrained emotion model
#model = tf.keras.models.load_model('./static/emoteSaved')
model = tf.keras.models.load_model('./static/emote.h5')

#detect emotion and put frame in place
def detect_emote(frame):
  frame=cv2.flip(frame,1)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
  for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
      roi_gray = gray[y:y + h, x:x + w]
      cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
      prediction = model.predict(cropped_img)
      maxindex = int(np.argmax(prediction))
      emote_text = emotion_dict[maxindex]
      
      cv2.putText(frame, emote_text, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
      
  return frame

#fetch frame and edit it to have emote


#instatiate flask app  
app = Flask(__name__, template_folder='./templates')

socketio = SocketIO(app,cors_allowed_origins='*' )

@app.route('/', methods=['POST', 'GET'])

def index():
    return render_template('index.html')


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string  = base64_string[idx+7:]
    sbuf = io.BytesIO()
    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)


    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)  

@socketio.on('image')
def image(data_image):  # generate frame by frame from camera
    #global fps,cnt, prev_recv_time,fps_array
    # raise Exception("Sorry, no numbers below zero")
    frame = readb64(data_image)
        
        
    frame= detect_emote(frame)
    #try:
    #frame=cv2.flip(frame,1)
    imgencode = cv2.imencode('.jpeg', frame,[cv2.IMWRITE_JPEG_QUALITY,40])[1]
    #encode it to string
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData
    # emit the frame back
    emit('response_back', stringData)
    
    
if __name__ == '__main__':
    socketio.run(app,port=5000 ,debug=True)
    
