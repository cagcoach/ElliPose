from typing import List

import numpy as np

from src.Skeleton.Keypoint import Position, Feature, Keypoint

_bones = [
        # body
        [Keypoint(Position.left, Feature.hip), Keypoint(Position.right, Feature.hip)],
        [Keypoint(Position.left, Feature.hip), Keypoint(Position.center, Feature.hip)],
        [Keypoint(Position.center, Feature.hip), Keypoint(Position.right, Feature.hip)],
        [Keypoint(Position.left, Feature.knee), Keypoint(Position.left, Feature.hip)],
        [Keypoint(Position.left, Feature.ancle), Keypoint(Position.left, Feature.knee)],
        [Keypoint(Position.right, Feature.knee), Keypoint(Position.right, Feature.hip)],
        [Keypoint(Position.right, Feature.ancle), Keypoint(Position.right, Feature.knee)],

        [Keypoint(Position.left, Feature.shoulder), Keypoint(Position.right, Feature.shoulder)],
        [Keypoint(Position.left, Feature.shoulder), Keypoint(Position.center, Feature.shoulder)],
        [Keypoint(Position.center, Feature.shoulder), Keypoint(Position.right, Feature.shoulder)],
        [Keypoint(Position.left, Feature.shoulder), Keypoint(Position.left, Feature.elbow)],
        [Keypoint(Position.left, Feature.elbow), Keypoint(Position.left, Feature.wrist)],
        [Keypoint(Position.right, Feature.shoulder), Keypoint(Position.right, Feature.elbow)],
        [Keypoint(Position.right, Feature.elbow), Keypoint(Position.right, Feature.wrist)],

        # head
        [Keypoint(Position.center, Feature.nose), Keypoint(Position.left, Feature.eye)],
        [Keypoint(Position.center, Feature.nose), Keypoint(Position.left, Feature.ear)],
        [Keypoint(Position.center, Feature.nose), Keypoint(Position.right, Feature.eye)],
        [Keypoint(Position.center, Feature.nose), Keypoint(Position.right, Feature.ear)],
        [Keypoint(Position.center, Feature.nose), Keypoint(Position.center, Feature.tophead)],

        [Keypoint(Position.left, Feature.eye), Keypoint(Position.left, Feature.ear)],
        [Keypoint(Position.left, Feature.eye), Keypoint(Position.right, Feature.eye)],
        [Keypoint(Position.left, Feature.eye), Keypoint(Position.right, Feature.ear)],
        [Keypoint(Position.left, Feature.eye), Keypoint(Position.center, Feature.tophead)],

        [Keypoint(Position.left, Feature.ear), Keypoint(Position.right, Feature.eye)],
        [Keypoint(Position.left, Feature.ear), Keypoint(Position.right, Feature.ear)],
        [Keypoint(Position.left, Feature.ear), Keypoint(Position.center, Feature.tophead)],

        [Keypoint(Position.right, Feature.eye), Keypoint(Position.right, Feature.ear)],
        [Keypoint(Position.right, Feature.eye), Keypoint(Position.center, Feature.tophead)],

        [Keypoint(Position.right, Feature.ear), Keypoint(Position.center, Feature.tophead)],
    ]

_drawBones = [
        [Keypoint(Position.left, Feature.hip), Keypoint(Position.right, Feature.hip)],
        [Keypoint(Position.left, Feature.hip), Keypoint(Position.center, Feature.hip)],
        [Keypoint(Position.center, Feature.hip), Keypoint(Position.right, Feature.hip)],
        [Keypoint(Position.left, Feature.knee), Keypoint(Position.left, Feature.hip)],
        [Keypoint(Position.left, Feature.ancle), Keypoint(Position.left, Feature.knee)],
        [Keypoint(Position.right, Feature.knee), Keypoint(Position.right, Feature.hip)],
        [Keypoint(Position.right, Feature.ancle), Keypoint(Position.right, Feature.knee)],

        [Keypoint(Position.left, Feature.shoulder), Keypoint(Position.left, Feature.hip)],
        [Keypoint(Position.right, Feature.shoulder), Keypoint(Position.right, Feature.hip)],

        [Keypoint(Position.left, Feature.shoulder), Keypoint(Position.right, Feature.shoulder)],
        [Keypoint(Position.left, Feature.shoulder), Keypoint(Position.center, Feature.shoulder)],
        [Keypoint(Position.center, Feature.shoulder), Keypoint(Position.right, Feature.shoulder)],
        [Keypoint(Position.left, Feature.shoulder), Keypoint(Position.left, Feature.elbow)],
        [Keypoint(Position.left, Feature.elbow), Keypoint(Position.left, Feature.wrist)],
        [Keypoint(Position.right, Feature.shoulder), Keypoint(Position.right, Feature.elbow)],
        [Keypoint(Position.right, Feature.elbow), Keypoint(Position.right, Feature.wrist)],

        [Keypoint(Position.left, Feature.shoulder), Keypoint(Position.center, Feature.nose)],
        [Keypoint(Position.right, Feature.shoulder), Keypoint(Position.center, Feature.nose)],
        [Keypoint(Position.center, Feature.shoulder), Keypoint(Position.center, Feature.nose)],

        # head
        [Keypoint(Position.center, Feature.nose), Keypoint(Position.left, Feature.eye)],
        [Keypoint(Position.center, Feature.nose), Keypoint(Position.right, Feature.eye)],
        [Keypoint(Position.center, Feature.nose), Keypoint(Position.center, Feature.tophead)],

        [Keypoint(Position.left, Feature.eye), Keypoint(Position.left, Feature.ear)],

        [Keypoint(Position.right, Feature.eye), Keypoint(Position.right, Feature.ear)],
]

class Bones:
    @staticmethod
    def getBonemat(columns, excludeKeypoints = None, bones = None):
        if bones is None:
            bones = _bones
        if excludeKeypoints == None:
            excludeKeypoints = []
        out = np.empty((0,len(columns)))
        boneslist = list()
        for row in bones:
            if row[0] in columns and row[1] in columns and row[0] not in excludeKeypoints and row[1] not in excludeKeypoints:
                newrow = np.zeros(len(columns))
                for i in range(len(columns)):
                    if columns[i] == row[0]:
                        newrow[i] += 1
                    if columns[i] == row[1]:
                        newrow[i] -= 1
                out = np.vstack((out,newrow))
                boneslist.append((row[0],row[1]))
        return out, boneslist

    @staticmethod
    def getDrawBones(availableBones: List[Keypoint]):
        l = list()
        for i in _drawBones:
            if i[0] in availableBones and i[1] in availableBones:
                l.append([availableBones.index(i[0]),availableBones.index(i[1])])
        return l