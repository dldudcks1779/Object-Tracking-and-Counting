class TrackableObject:
    def __init__(self, objectID, centroid):
        # 객체 ID
        self.objectID = objectID

        # 객체 중심 좌표 목록
        self.centroids = [centroid]
        
        # 객체 counting 여부
        self.counted = False
