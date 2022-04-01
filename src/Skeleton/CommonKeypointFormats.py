from src.Skeleton.Keypoint import Feature, Position, Keypoint

class COMMON_KEYPOINT_FORMATS:
    @staticmethod
    def coco():
        return [Keypoint(Position.center, Feature.nose),  # 0
         Keypoint(Position.left, Feature.eye),  # 1
         Keypoint(Position.right, Feature.eye),  # 2
         Keypoint(Position.left, Feature.ear),  # 3
         Keypoint(Position.right, Feature.ear),  # 4
         Keypoint(Position.left, Feature.shoulder),  # 5
         Keypoint(Position.right, Feature.shoulder),  # 6
         Keypoint(Position.left, Feature.elbow),  # 7
         Keypoint(Position.right, Feature.elbow),  # 8
         Keypoint(Position.left, Feature.wrist),  # 9
         Keypoint(Position.right, Feature.wrist),  # 10
         Keypoint(Position.left, Feature.hip),  # 11
         Keypoint(Position.right, Feature.hip),  # 12
         Keypoint(Position.left, Feature.knee),  # 13
         Keypoint(Position.right, Feature.knee),  # 14
         Keypoint(Position.left, Feature.ancle),  # 15
         Keypoint(Position.right, Feature.ancle),  # 16
         ]

    @staticmethod
    def unknownWithMHipAndTopHead():
        return [Keypoint(Position.center, Feature.hip),  # 0
         Keypoint(Position.right, Feature.hip),  # 1
         Keypoint(Position.right, Feature.knee),  # 2
         Keypoint(Position.right, Feature.ancle),  # 3
         Keypoint(Position.left, Feature.hip),  # 4
         Keypoint(Position.left, Feature.knee),  # 5
         Keypoint(Position.left, Feature.ancle),  # 6
         Keypoint(Position.center, Feature.shoulder),  # 7
         Keypoint(Position.center, Feature.neck),  # 8
         Keypoint(Position.center, Feature.nose),  # 9
         Keypoint(Position.center, Feature.tophead),  # 10
         Keypoint(Position.left, Feature.shoulder),  # 11
         Keypoint(Position.left, Feature.elbow),  # 12
         Keypoint(Position.left, Feature.wrist),  # 13
         Keypoint(Position.right, Feature.shoulder),  # 14
         Keypoint(Position.right, Feature.elbow),  # 15
         Keypoint(Position.right, Feature.wrist),  # 16
         ]