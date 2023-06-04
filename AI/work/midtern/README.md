**本程式碼為原創**



程式碼:

* [buildDataSet.py](buildDataSet.py): 建置資料集，裡面要自訂義目前的face box要對應到哪幾個face mesh的點，有提供資料集data.csv、data_label.csv
* [ModelToTFlite.py](ModelToTFlite.py): 訓練出來的模型轉換成tflite，這很重要，因為速度差很多
* [useModel](useModel.py): 使用模型，運行face box模擬face mesh，這裡有提供訓練完的模型DNNFaceModel.tflite



[mediapipe](https://arxiv.org/pdf/1906.08172.pdf)是google開發的影像辨識套件，可以增測人體座標點，這裡針對兩個mediapipe的模型做探討: face box和face mesh，face box有良好的效能，能夠偵測到人臉較少部分，而face mesh則是有許多點，一共有468個，所以face mesh比face box消耗的效能還要高，為了節省效能，我開發了一個利用face box推測face mesh的DNN神經網路。



# DNN structure

使用 face mesh 產生的穴位點座標資訊以及 face box 的座標點資訊 來生成訓練集，然後使用深度神經網路（DNN）進行 150 個 epochs 的訓練，其中每個 batch 的大小為16。在訓練過程中，使用均方誤差（MSE）作為損失函數。![](picture/DNN_structure.png)



# DNN metric

可以看到 DNN 頭部模型的準確度達到了高達九成



![](picture/DNN_loss.png)

# Demo

可以看到，基本上可以取代Face mesh

![](picture/face.gif)