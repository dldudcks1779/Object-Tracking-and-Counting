<div>
  <p align="center">
    <img width="700" src="result_video.gif">
  </p>
</div>

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
## Centroid Tracking Algorithm
* #### 1. Object Detection을 통해 단일 프레임에서 감지된 각 객체에 대한 Bounding Box의 좌표를 수집한 후 중심 좌표를 계산
* #### 2. 각 중심 좌표에 고유 ID를 할당하고 객체가 움직이면서 갱신되는 새로운 프레임의 중심 좌표와 기존 프레임의 중심 좌표 사이의 유클리드 거리(Euclidean Distance)를 계산
  * ##### 유클리드 거리(Euclidean Distance) : 두 점 사이의 거리
* #### 3. 기존 프레임의 중심 좌표와 새로운 프레임의 중심 좌표 사이의 유클리드 거리가 가장 가까운 두 쌍은 동일 객체라고 판단하고 기존의 ID를 할당
  * ##### 허용하는 최대 거리 지정
* #### 4. ID가 할당되지 않은 중심 좌표에는 새로운 객체라고 판단하여 고유한 ID를 할당

* #### 예)

<div>
  <p align="center">
    <img width="500" src="Centroid Tracking Algorithm.png">
  </p>
</div>

  * #### 현재 프레임(F1)에서 Object Detection을 통해 객체의 중심 좌표를 계산한 후 각 중심 좌표에 고유 ID(ID_1, ID_1)를 할당
  * #### 다음 프레임(F2)에서도 Object Detection을 통해 객체의 중심 좌표를 계산한 후 현재 프레임의 중심 좌표와 다음 프레임의 중심 좌표 사이의 유클리드 거리를 각각 계산
  * #### 현재 프레임과 다음 프레임의 중심 좌표 사이의 유클리드 거리가 가장 가까운 두 쌍은 동일 객체라고 판단하여 그 중심 좌표에는 기존의 ID(ID_1, ID_2)를 할당
  * #### 다음 프레임에서 ID가 할당되지 않은 중심 좌표에는 새로운 객체라고 판단하여 고유한 ID(ID_3)를 할당
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
---
## YOLO 객체 추적 및 카운팅 시스템(YOLO Object Tracking and Counting System)
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

---
