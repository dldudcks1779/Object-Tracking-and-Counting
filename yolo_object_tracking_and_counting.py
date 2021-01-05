##### 실행 #####
# 비디오를 저장하지 않을 경우
# webcam : sudo python3 yolo_object_tracking_and_counting.py
# 예) sudo python3 yolo_object_tracking_and_couning.py
# video : sudo python3 yolo_object_tracking_and_couning.py --input 비디오 경로
# 예) sudo python3 yolo_object_tracking_and_couning.py --input test_video.mp4
#
# 비디오를 저장할 경우
# webcam : sudo python3 yolo_object_tracking_and_couning.py --output 저장할 비디오 경로
# 예) sudo python3 yolo_object_tracking_and_couning.py --output result_video_yolo.avi
# video : sudo python3 yolo_object_tracking_and_couning.py --input 비디오 경로 --output 저장할 비디오 경로
# 예) sudo python3 yolo_object_tracking_and_couning.py --input test_video.mp4 --output result_video_yolo.avi

# 필요한 패키지 import
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import argparse # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import imutils # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
import time # 시간 처리 모듈
import dlib # 이미지 처리 및 기계 학습, 얼굴인식 등을 할 수 있는 고성능의 라이브러리
import cv2 # opencv 모듈
import os # 운영체제 기능 모듈

# 실행을 할 때 인자값 추가
ap = argparse.ArgumentParser() # 인자값을 받을 인스턴스 생성
# 입력받을 인자값 등록
ap.add_argument("-i", "--input", type=str, help="input 비디오 경로")
ap.add_argument("-o", "--output", type=str, help="output 비디오 경로") # 비디오 저장 경로
ap.add_argument("-c", "--confidence", type=float, default=0.7, help="최소 확률")
ap.add_argument("-s", "--skip-frames", type=int, default=30, help="추적된 객체에서 다시 객체를 탐지하기까지 건너뛸 프레임 수")
# 입력받은 인자값을 args에 저장
args = vars(ap.parse_args())

# YOLO 모델이 학습된 coco 클래스 레이블
labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# YOLO 가중치 및 모델 구성에 대한 경로
weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"]) # 가중치
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"]) # 모델 구성

# COCO 데이터 세트(80 개 클래스)에서 훈련된 YOLO 객체 감지기 load
print("[YOLO 객체 감지기 loading...]")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# YOLO에서 필요한 output 레이어 이름
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# input 비디오 경로가 제공되지 않은 경우 webcam
if not args.get("input", False):
    print("[webcam 시작]")
    vs = cv2.VideoCapture(0)

# input 비디오 경로가 제공된 경우 video
else:
    print("[video 시작]")
    vs = cv2.VideoCapture(args["input"])

# 비디오 저장 변수 초기화
writer = None

# 프레임 크기 초기화(비디오에서 첫 번째 프레임을 읽는 즉시 설정)
W = None
H = None

# 중심 추적 변수
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)

# 추적 객체 목록
trackers = []

# 추적 객체 ID
trackableObjects = {}

# 총 프레임 수
totalFrames = 0 # 총 프레임 수

# 총 이동 객체 수
totalRight = 0
totalLeft = 0

# fps 정보 초기화
fps = FPS().start()

# 객체 시작점 튜플
object_start_tuple = ()

# 비디오 스트림 프레임 반복
while True:
    # 프레임 읽기
    ret, frame = vs.read()
    
    # 읽은 프레임이 없는 경우 종료
    if args["input"] is not None and frame is None:
        break
    
    # 프레임 크기 지정
    frame = imutils.resize(frame, width=1000)
    
    # RGB 변환
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 프레임 크기
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
    # output video 설정
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
    
    # 객체 bounding box 목록
    rects = []

    # 추적된 객체에서 다시 객체를 탐지하기까지 건너뛸 프레임 수 적용
    # 객체를 탐지할 때
    if totalFrames % args["skip_frames"] == 0:
        # 객체 추적 목록 초기화
        trackers = []

        # blob 이미지 생성
        # 파라미터
        # 1) image : 사용할 이미지
        # 2) scalefactor : 이미지 크기 비율 지정
        # 3) size : Convolutional Neural Network에서 사용할 이미지 크기를 지정
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # 객체 인식
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # layerOutputs 반복
        for output in layerOutputs:
            # 각 클래스 레이블마다 인식된 객체 수 만큼 반복
            for detection in output:
                # 인식된 객체의 클래스 ID 및 확률 추출
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                
                # 사람인 경우
                if classID == 0: # yolo-coco 디렉터리에 coco.names 파일을 참고하여 다른 object 도 인식 가능(0 인 경우 사람)
                    # 객체 확률이 최소 확률보다 큰 경우
                    if confidence > args["confidence"]:
                        # bounding box 위치 계산
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int") # (중심 좌표 X, 중심 좌표 Y, 너비(가로), 높이(세로))

                        # bounding box  좌표
                        startX = int(centerX - (width / 2))
                        startY = int(centerY - (height / 2))
                        endX = int(centerX + (width / 2))
                        endY = int(centerY + (height / 2))
            
                        # 객체 추적 정보 추출
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(rgb, rect)
                
                        # 인식된 객체를 추적 목록에 추가
                        trackers.append(tracker)

    # 객체를 탐지하지 않을 때
    else:
        # 추적된 객체 수 만큼 반복
        for tracker in trackers:
            # 추적된 객체 위치
            tracker.update(rgb)
            pos = tracker.get_position()

            # bounding box 좌표 추출
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # bounding box 좌표 추가
            rects.append((startX, startY, endX, endY))
            
            # bounding box 출력
            # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Counting Line
    cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 0, 255), 2)
    cv2.putText(frame, "Counting Line", ((W // 2) + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 객체 중심 추적
    objects = ct.update(rects)

    # 추적된 객체 수 만큼 반복
    for (objectID, centroid) in objects.items():
        # 현재 객체 ID에 대해 추적 가능한 객체 확인
        to = trackableObjects.get(objectID, None)
        
        # 추적 가능한 객체가 없는 경우
        if to is None:
            # 하나의 객체를 생성
            to = TrackableObject(objectID, centroid)

        # 추적 가능한 객체가 있는 경우
        else:
            # 이전의 중심 좌표에 대한 가로 좌표 값을 추출
            y = [c[0] for c in to.centroids]

            # 현재 중심 좌표와 이전 중심 좌표의 평균의 차이를 이용하여 방향을 계산
            direction = centroid[0] - np.mean(y)

            # 중심 좌표 추가
            to.centroids.append(centroid)

            # 객체가 counting 되지 않았을 때
            if not to.counted:
                # 객체가 왼쪽에서 시작
                if centroid[0] < (W // 2) - 20:
                    try:
                        if len(object_start_tuple) < objectID:
                            object_start_tuple = object_start_tuple + (0,)
                        elif len(object_start_tuple) == objectID:
                            object_start_tuple = object_start_tuple + (-1,)
                    except:
                        pass
                
                # 객체가 오른쪽에서 시작
                elif centroid[0] > (W // 2) + 20:
                    try:
                        if len(object_start_tuple) < objectID:
                            object_start_tuple = object_start_tuple + (0,)
                        elif len(object_start_tuple) == objectID:
                            object_start_tuple = object_start_tuple + (1,)
                    except:
                        pass
                
                try:
                    # 객체가 왼쪽으로 이동
                    if object_start_tuple[objectID] == 1 and direction < 0 and centroid[0] < W // 2:
                        to.counted = True
                        totalLeft += 1
                        print("Left")
                    
                    # 객체가 오른쪽으로 이동
                    elif object_start_tuple[objectID] == -1 and direction > 0 and centroid[0] > W // 2:
                        to.counted = True
                        totalRight += 1
                        print("Right")
                except:
                    print("error")
                    pass

        # 추적 가능한 객체 저장
        trackableObjects[objectID] = to
        
        # 객체 ID
        # text = "ID {}".format(objectID)

        # 객체 ID 출력
        # cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 객체 중심 좌표 출력
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -2)

    # Counting 정보
    info = [
            ("Left", totalLeft),
            ("Right", totalRight),
    ]

    # Counting 정보를 반복
    for (i, (k, v)) in enumerate(info):
        # Counting 정보
        text = "{} : {}".format(k, v)

        # Counting 정보 출력
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 비디오 저장
    if writer is not None:
        writer.write(frame)
    
    # 프레임 출력
    cv2.imshow("People Tracking and Counting", frame)
    key = cv2.waitKey(1) & 0xFF

    # 'q' 키를 입력하면 종료
    if key == ord("q"):
        break
    
    # 총 프레임 수 증가
    totalFrames += 1
    
    # fps 정보 업데이트
    fps.update()

# fps 정지 및 정보 출력
fps.stop()
print("[재생 시간 : {:.2f}초]".format(fps.elapsed()))
print("[FPS : {:.2f}]".format(fps.fps()))

# 비디오 저장 종료
if writer is not None:
    writer.release()

# 종료
vs.release()
cv2.destroyAllWindows()
