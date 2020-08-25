# 필요한 패키지 import
from scipy.spatial import distance as dist # 거리 계산
from collections import OrderedDict # 딕셔너리 순서대로 사용
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈

# 중심 추적
class CentroidTracker:
    # 생성자
    def __init__(self, maxDisappeared=40, maxDistance=50):
        # 객체 고유 ID 초기화
        self.nextObjectID = 0
        
        # 객체 ID와 중심 좌표 딕셔너리
        self.objects = OrderedDict()
        
        # 특정 객체가 사라짐으로 표시된 프레임 수
        self.disappeared = OrderedDict()
        
        # 객체가 사라짐으로 표시될 수 있는 최대 프레임 수
        self.maxDisappeared = maxDisappeared
        
        # 최대 거리
        self.maxDistance = maxDistance

    # 새로운 객체 등록
    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid # 고유 ID 할당 및 객체 중심 좌표 등록
        self.disappeared[self.nextObjectID] = 0 # 객체가 사라진 프레임 수를 0으로 초기화
        self.nextObjectID += 1 # 다음 객체 ID
    
    # 사라진 객체 삭제
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
    
    # 객체 중심 좌표 추적
    def update(self, rects):
        # bounding box 목록이 비어있는 경우
        if len(rects) == 0:
            # 기존의 추적된 객체를 반복하여 사라진 것으로 표시
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1 # 사라진 프레임 수 증가
                
                # 객체가 사라졌다고 판단한 프레임 수가 최대 프레임 수 보다 클 경우
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID) # 사라진 객체 삭제

            # 업데이트할 추적 정보가 없는 경우 반환
            return self.objects
        
        # 현재 프레임에 대한 중심 좌표 목록 초기화
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        
        # 중심 좌표 목록만큼 반복
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # 중심 좌표 계산
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            
            # 중심 좌표 목록에 저장
            inputCentroids[i] = (cX, cY)
        
        # 추적하는 객체가 없는 경우
        if len(self.objects) == 0:
            # 중심 좌표 길이만큼 반복
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i]) # 새로운 객체 등록
                
        # 추적하는 객체가 있는 경우
        else:
            # 객체 ID
            objectIDs = list(self.objects.keys())

            # 객체 중심 좌표
            objectCentroids = list(self.objects.values())
            
            # 현재 프레임의 객체 중심 좌표와 이전 프레임의 객체 중심 좌표 사이의 거리를 계산
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            
            # 각 행에서 가장 작은 값을 찾은 다음 오름차순 정렬
            rows = D.min(axis=1).argsort()
            
            # 각 열에서 가장 작은 값을 찾은 다음 행을 기준으로 정렬
            cols = D.argmin(axis=1)[rows]
            
            # 행과 열 인덱스를 사용했는지 확인하기 위한 집합
            usedRows = set()
            usedCols = set()
            
            # (행, 열) 인덱스 튜블 조합만큼 반복
            for (row, col) in zip(rows, cols):
                # 행 또는 열 값을 이미 사용한 경우
                if row in usedRows or col in usedCols:
                    continue
                
                # 중심 좌표 사이의 거리가 최대 거리 보다 큰 경우
                if D[row, col] > self.maxDistance:
                    continue
                
                # 객체의 중심 좌표 업데이트
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0 # 객체가 사라진 프레임 수 초기화
                
                # 행과 열 인덱스 사용한 경우 각 집합에 추가
                usedRows.add(row)
                usedCols.add(col)
            
            # 사용하지 않은 행과 열 인덱스를 모두 계산하고 저장
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            # 이전 프레임의 객체 중심 좌표 수가 현재 프레임의 객체 중심 좌표 수 보다 크거나 같은 경우
            if D.shape[0] >= D.shape[1]:
                # 사용하지 않은 행 인덱스 반복
                for row in unusedRows:
                    # 해당 행 인덱스에 대한 객체 ID
                    objectID = objectIDs[row]
                    
                    # 사라진 프레임 수 증가
                    self.disappeared[objectID] += 1
                    
                    # 객체가 사라졌다고 판단한 프레임 수가 최대 프레임 수 보다 클 경우
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID) # 사라진 객체 삭제
            
            # 이전 프레임의 객체 중심 좌표 수가 현재 프레임의 객체 중심 좌표 수 보다 작은 경우
            else:
                # 사용하지 않은 열 인덱스 반복
                for col in unusedCols:
                    self.register(inputCentroids[col]) # 새로운 객체 등록
        
        # 추적 가능한 객체 딕셔너리 반환
        return self.objects
