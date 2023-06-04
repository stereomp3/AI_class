import cv2
import mediapipe as mp  # 載入 mediapipe 函式庫
import numpy as np
from mediapipe.framework.formats import landmark_pb2

cap = cv2.VideoCapture(0)
mp_face_detection = mp.solutions.face_detection  # 建立偵測方法
mp_drawing = mp.solutions.drawing_utils  # 建立繪圖方法
mp_face_mesh = mp.solutions.face_mesh

acupuncture = []
point = [10, 164, 425, 417, 298, 266, 451, 229, 287, 435, 280, 454, 464, 338, 297]

# fw = open('./data.csv', 'w')
fw = open('./data.csv', 'a')

def set_landmark_point(landmark):
    landmark_subset = landmark_pb2.NormalizedLandmarkList(
        landmark=[landmark[point[i]] for i in range(len(point))]
    )
    return landmark_subset

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
        mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:  # 開始偵測人臉

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
        results_face_mesh = face_mesh.process(img2)

        if results.detections and results_face_mesh.multi_face_landmarks:
            for detection in results.detections:
                mp_drawing.draw_detection(img, detection)  # 標記人臉
                for face_point in detection.location_data.relative_keypoints:
                    pass
                    # print(face_point.x)
                    # fw.write(str(round(face_point.x, 4)) + "," + str(round(face_point.y, 4)) + ",")

                for face_landmarks in results_face_mesh.multi_face_landmarks:
                    landmark_subset = set_landmark_point(face_landmarks.landmark)
                    mp_drawing.draw_landmarks(image=img, landmark_list=landmark_subset)
                    count = 0
                    for landmark in landmark_subset.landmark:
                        count += 1
                        # if count <= len(point) - 1:
                        #     fw.write(str(round(landmark.x, 4)) + "," + str(round(landmark.y, 4)) + ",")
                        # else:
                        #     fw.write(str(round(landmark.x, 4)) + "," + str(round(landmark.y, 4)) + "\n")




        cv2.imshow('build_data', img)
        if cv2.waitKey(5) == ord('q'):
            break  # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()
fw.close()