import copy
from typing import List

import numpy as np
from src.Skeleton.Keypoint import Keypoint, Position, Feature


class Skeleton:
    '''
    usage: Skeleton(keypoints:np.array, annotations:list(str))
           Skeleton(dict<annotation, list(keypoints)>
    '''
    _keypoints = dict()

    def __init__(self, skeleton:"Skeleton"):
        self._keypoints = copy.deepcopy(skeleton._keypoints)

    def __init__(self, ids:list, nparray:np.array):
        assert len(ids) == nparray.shape[0]
        for i in range (len(ids)):
            self.addKeypoint(Keypoint(keypointID=ids[i], vec=nparray[i]))

    def addKeypoint(self, kp : Keypoint):
        self._keypoints[kp.getKeypointID()] = kp

    def getKeypointsAsNpArray(self, ids:list = None) -> np.array:
        if ids is None:
            return np.vstack([k.vec for k in self._keypoints]), self._keypoints.keys()
        else:
            return np.vstack([self._keypoints[id].vec for id in ids]), ids

    def getMirroredSkeleton(self) -> "Skeleton":
        out = Skeleton(self)
        for kp in out._keypoints.values():
            kp.mirrorAnnotations()

    def getAvailableKeypointIDs(self) -> List[int]:
        return self._keypoints.keys()