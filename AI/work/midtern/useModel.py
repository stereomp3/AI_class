import cv2
import mediapipe as mp
# Load the saved model
import mediapipe.framework.formats.landmark_pb2
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
from mediapipe.framework.formats import landmark_pb2

# Memory leak when repeatedly loading and deleting keras models: https://github.com/tensorflow/tensorflow/issues/40171
gpus = tf.config.experimental.list_physical_devices('GPU')
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    except RuntimeError as e:
        print(e)

loaded_model = tf.lite.Interpreter(model_path='DNNFaceModel.tflite')
loaded_model.allocate_tensors()


def get_predictions(landmark, hasData=False):
    new_data = []
    if not hasData:
        for l in landmark:
            new_data.append(l.x)
            new_data.append(l.y)
    else:
        new_data = landmark
    # Get input and output tensors
    input_data = np.array([new_data], dtype=np.float32)
    input_details = loaded_model.get_input_details()
    output_details = loaded_model.get_output_details()
    loaded_model.set_tensor(input_details[0]['index'], input_data)
    loaded_model.invoke()
    # results = loaded_model.predict([new_data])[0]
    results = loaded_model.get_tensor(output_details[0]['index'])[0]
    return results


new_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

get_predictions(new_data, hasData=True)


# dataaa = {'x': 0.1, 'y': 0.1, 'z': 0.1}


def set_landmark_point(landmark):
    output = get_predictions(landmark)
    print(landmark[0])
    landmark_subset = landmark_pb2.NormalizedLandmarkList(
        landmark=[{'x': output[i], 'y': output[i + 1], 'z': 0} for i in range(0, len(output), 2)]
    )
    return landmark_subset


# print(set_landmark_point(new_data))

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils  # 建立繪圖方法
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:  # 開始偵測人臉

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將 BGR 顏色轉換成 RGB
        size = img.shape  # 取得攝影機影像尺寸
        w = size[1]  # 取得畫面寬度
        h = size[0]  # 取得畫面高度
        results = face_detection.process(img2)  # 偵測人臉

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img, detection)  # 標記人臉
                acupuncture_landmark = set_landmark_point(detection.location_data.relative_keypoints)
                mp_drawing.draw_landmarks(image=img, landmark_list=acupuncture_landmark)

        cv2.imshow('build_data', img)
        if cv2.waitKey(5) == ord('q'):
            break  # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()
