## MobileNet SSD
* #### 빠르고 효율적인 딥러닝 기반 Object Detection을 위해 MobileNet과 SSD를 결합한 MobileNet SSD를 사용
* #### 이미지를 분류하기 위한 CNN(Convolution Neural Network)의 MobileNet은 같은 레이어 수의 다른 CNN 구조에 비해 낮은 파라미터 수로 인하여 작은 응답 지연 시간을 가짐
* #### 객체 인식을 위한 딥러닝 모델 중 하나인 SSD(Single Shot Multibox Detector)는 입력한 하나의 이미지만 CNN을 실행하여 객체를 탐지
  * ##### caffemodel 파일 : Object Detection을 위해 사전 훈련된 모델인 MobileNet SSD caffemodel을 사용(약 20개의 객체 인식)
  * ##### prototxt 파일 : 모델의 레이어 구성 및 속성 정의
---
## YOLOv3(You Only Look Once v3)
* #### grid cell로 나누어 한 번에 클래스를 판단하고 통합하여 최종 객체를 판단
* #### Bounding Box Coordinate(좌표) 및 클래스 Classification(분류)을 동일 신경망 구조를 통해 동시에 실행
* #### 사람, 자전거, 자동차, 개, 고양이 등 약 80개의 레이블로 구성
  * ##### yolov3.weights 파일 : 사전 훈련된 네트워크 가중치
    * ##### 다운로드 : https://drive.google.com/drive/folders/1QnZHzsss3Jdz2QhvF3CKu0avBF7eMhlV?usp=sharing
  * ##### yolov3.cfg 파일 : 네트워크 구성
  * ##### coco.names 파일 : coco dataset에 사용된 80가지 클래스 이름
---
### 실행 환경
* #### Ubuntu
* #### OpenCV Version : 3.x.x
  * ##### 설치 : https://blog.naver.com/dldudcks1779/222020005648
* #### dlib
  * ##### 설치 : https://blog.naver.com/dldudcks1779/222024194834
* #### imutils
  * ##### 설치 : sudo pip3 install imutils
---
## 객체 추적 및 카운팅 시스템(Object Tracking and Counting System)
* #### 비디오를 저장하지 않을 경우
  * webcam : sudo python3 object_tracking_and_counting.py
    * 예) ssudo python3 object_tracking_and_couning.py
  * video : sudo python3 object_tracking_and_couning.py --input 비디오 경로
    * 예) sudo python3 object_tracking_and_couning.py --input test_video.mp4
* #### 비디오를 저장할 경우
  * webcam : sudo python3 object_tracking_and_couning.py --output 저장할 비디오 경로
    * 예) sudo python3 object_tracking_and_couning.py --output result_video.avi
  * video : sudo python3 object_tracking_and_couning.py --input 비디오 경로 --output 저장할 비디오 경로
    * 예) sudo python3 object_tracking_and_couning.py --input test_video.mp4 --output result_video.avi

<div>
  <p align="center">
    <img width="300" src="result_video/result_video_1.gif">
  </p>
</div>

---
## 객체 추적 및 카운팅 시스템(Object Tracking and Counting System)
* #### 비디오를 저장하지 않을 경우
  * webcam : sudo python3 yolo_object_tracking_and_counting.py
    * 예) sudo python3 yolo_object_tracking_and_couning.py
  * video : sudo python3 yolo_object_tracking_and_couning.py --input 비디오 경로
    * 예) sudo python3 yolo_object_tracking_and_couning.py --input test_video.mp4
* #### 비디오를 저장할 경우
  * webcam : sudo python3 yolo_object_tracking_and_couning.py --output 저장할 비디오 경로
    * 예) sudo python3 yolo_object_tracking_and_couning.py --output result_video_yolo.avi
  * video : sudo python3 yolo_object_tracking_and_couning.py --input 비디오 경로 --output 저장할 비디오 경로
    * 예) sudo python3 yolo_object_tracking_and_couning.py --input test_video.mp4 --output result_video_yolo.avi
<div>
  <p align="center">
    <img width="300" src="result_video/result_video_3.gif">
  </p>
</div>
---
