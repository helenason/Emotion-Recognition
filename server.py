import cv2
import numpy as np
import tensorflow as tf
import json
import base64
from flask import Flask, render_template, Response, jsonify
from music_list import Music

app = Flask(__name__)

first_emotion = []
second_emotion = []
emotion_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

yolonet=cv2.dnn.readNet('yoloface-master/model-weights/yolov3-wider_16000.weights','yoloface-master/cfg/yolov3-face.cfg')

# model = tf.keras.models.load_model('facial_expression_model.h5')
# model = tf.keras.models.load_model('facial_expression_with_layers_mj.h5')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), padding='same', input_shape=(48,48,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(256, (5,5), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(512, (3,3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(1024, (3,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(4096),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(7, activation='softmax')
])

with tf.device('/device:GPU:0'):
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])

model.load_weights('facial_expression.h5')


layer_names = yolonet.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolonet.getUnconnectedOutLayers()]

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)

# 카메라 size 설정
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
channel = 3

colors = np.random.uniform(0,255,size=7) 
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.startWindowThread()

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    # cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()  # 프레임 읽기

        # if cv2.waitKey(1)==ord('q'):
        #     break
        if ret is None:
            print('ret')
        if frame is None:
            print('frame')

        # preprocessing for yolo
        blob=cv2.dnn.blobFromImage(frame,0.00392,(416,416),(0,0,0),True,crop=False)
        yolonet.setInput(blob)
        outs=yolonet.forward(output_layers)
        
        boxes=[]
        confidences=[]
        class_ids=[]
        for out in outs:
            for detection in out:
                confidence=detection[5] #class face로 1개이므로 [5:]가 아닌 5만 보면 됨
                if confidence>0.1: #정확도를 위해서 올릴 수 있어
                    center_x=int(detection[0]*width)
                    center_y=int(detection[1]*height)
                    w=int(detection[2]*width)
                    h=int(detection[3]*height)

                    x=int(center_x-w/2)
                    y=int(center_y-h/2)

                    boxes.append([x,y,w,h]) #후보군들
                    confidences.append(float(confidence))
        
        indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.1,0.4)
        #boxes로 선정된 것 중 Confidence가 threshold이상인 것들만 indexes에

        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h=boxes[i]
                # bounding box 설정
                x_1=np.max([0,x-int(0.1*w)])
                y_1=np.max([0,y-int(0.1*h)])
                x_2=np.min([width,x+int(1.2*w)])
                y_2=np.min([height,y+int(1.1*h)])
                croped=frame[y_1:y_2,x_1:x_2]

                croped=cv2.resize(croped,(48, 48))/255.0
                croped=croped.reshape(1,48, 48, 3)

                y_pred=model.predict(croped,verbose=False)[0]
                #print('y_pred: ', y_pred)
                y_pred_softmax = tf.nn.softmax(y_pred)

                #print('y_pred_softmax: ', y_pred_softmax)
                top_class = tf.argmax(y_pred_softmax)
                first_emotion.append(int(top_class))
                #print('y_pred: ', y_pred)
                # 가장 높은 클래스와 확률 값 추출
                top_probability = y_pred_softmax[top_class]

                # 두 번째로 높은 클래스와 확률 값 추출
                sorted_indices = np.argsort(y_pred_softmax)[::-1]
                # 확률 값에 대해 내림차순으로 정렬된 인덱스
                second_class = sorted_indices[1]  # 두 번째로 높은 클래스
                second_emotion.append(int(second_class))
                second_probability = y_pred_softmax[second_class]

                cv2.rectangle(frame,(x_1,y_1),(x_2,y_2),colors[0],2)
                
                cv2.putText(frame,"Pred1: " + emotion_list[int(top_class)] + f"({round(int(top_probability * 100), 2)}%)" ,(40,80),font,1,colors[0],1)
                cv2.putText(frame,"Pred2: " + emotion_list[int(second_class)] + f"({round(int(second_probability * 100), 2)}%)",(40, 120),font,1,colors[0],1)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/music_list')
def music_list():
    # print(first_emotion[-1])
    # print(second_emotion[-1])
    first_res = first_emotion[-1]
    second_res = second_emotion[-1]

    result = Music.print_music(first_res, second_res)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
