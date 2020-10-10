#coding:utf-8
import cv2
import os
# 调用打开摄像头
cap = cv2.VideoCapture(0)

# 加载视频
#cap = cv2.VideoCapture('')填入路径

width = cap.get(3)#输出视频的宽度
high = cap.get(4)#输出视频的高度
print(width,high)

#导入特征分类器文件识别特征
#将haarcascade_frontalface_default.xml文件与find_face.py放在同一目录下即可
path = os.path.abspath(os.path.dirname(__file__))
face_detect = cv2.CascadeClassifier(path+"\haarcascade_frontalface_default.xml")
while True:
    # 读取视频片段
    flag, frame = cap.read()
    if flag == False:
        print("失败，请检查摄像头或者视频文件！")
        break

    # 灰度处理
    gray = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)

    # 检查人脸
    face_zone = face_detect.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)

    # 将检测到的人脸标识出来
    num = 0
    for x, y, w, h in face_zone:
        num = num + 1
        cv2.rectangle(frame, pt1 = (x, y), pt2 = (x+w, y+h), color = [0,0,255], thickness=2)   #画正方形
        #cv2.circle(frame, center = (x + w//2, y + h//2), radius = w//2, color = [0,255,0], thickness = 2)   #画圆
        cv2.putText(frame, str(num), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    # 显示图片
    cv2.putText(frame, "people:{}".format(num), (5,20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), )

    #自定义视频输出宽和高
    #cv2.namedWindow("video",0)
    #cv2.resizeWindow("video",W,H) #自定义视频输出宽和高
    cv2.imshow('video', frame)

    # 设置Esc键退出
    key = cv2.waitKey(10)
    if key  == 27:
        break
cap.release()

# 释放资源
cv2.destroyAllWindows()
