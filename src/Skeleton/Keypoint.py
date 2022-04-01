import math

import numpy as np
from enum import Enum

class Position(Enum):
    left = 1
    right = 2
    center = 0


class Feature(Enum):
    nose = 0
    eye = 1
    ear = 2
    tophead = 3
    neck = 4
    shoulder = 5
    elbow = 6
    wrist = 7
    hip = 8
    knee = 9
    ancle = 10

class Dimension(Enum):
    x=0
    y=1
    z=2
    accuracy=3
    visibility=4

class Keypoint:

    @staticmethod
    def from_name(name: str) -> "Keypoint":
        nsplit = name.split("_")
        return Keypoint(Position[nsplit[0].lower()],Feature[nsplit[1].lower()])

    HEAD = [Feature.nose, Feature.eye, Feature.ear, Feature.tophead]
    def __init__(self, kp_id:int):
        position = id % 3
        feature = math.floor(id / 3)
        self.pos = Position(position)
        self.feat = Feature(feature)

    def __init__(self, pos:Position, feat:Feature):
        self.pos = pos
        self.feat = feat

    def __str__(self):
        return "Keypoint(" + self.name + ")"

    def __repr__(self):
        return "Keypoint(" + self.name + ")"

    @property
    def id(self):
        return self.feat.value * 3 + self.pos.value

    @property
    def name(self):
        return str(self.pos.name) + "_" + str(self.feat.name)

    @property
    def isHead(self):
        return (self.feat in Keypoint.HEAD)

    def mirror(self):
        if self.pos == Position.right:
            self.pos = Position.left
        elif self.pos == Position.left:
            self.pos = Position.right

    def __eq__(self, other):
        if type(other) == Keypoint:
            return (self.feat == other.feat and self.pos == other.pos)
        if type(other) == int:
            return (self.id == other)
        if type(other) == tuple:
            return (self.feat == other[1] and self.pos == other[0])
        raise TypeError("Cannot compare with this type", type(other))
    def __hash__(self):
        return self.id
'''
class Keypoint:
    #def __init__(self, feature:Feature, position:Position, x:float, y:float, z:float = None):
    def __init__(self, kp: "Keypoint") -> "Keypoint":
        self.position = kp.position
        self.vec = kp.vec
        self.feature = kp.feature

    def __init__(self, **kwargs):

        #Set Keypoint Definition
        if "position" in kwargs and "feature" in kwargs:
            self.position = kwargs["position"]
            self.feature = kwargs["feature"]
        elif "keypointID" in kwargs:
            self.setKeypointID(kwargs["keypointID"])
        else:
            raise ValueError

        #Set Keypoint Position
        if "x" in kwargs and "y" in kwargs:
            if "z" in kwargs:
                self.vec = np.array([kwargs["x"], kwargs["y"], kwargs["z"]])
            else:
                self.vec = np.array([kwargs["x"], kwargs["y"]])
        elif "vec" in kwargs and len(kwargs["vec"].shape) == 1 and (kwargs["vec"].shape[0] == 2 or kwargs["vec"].shape[0] == 3):
            self.vec = kwargs["vec"]
        else:
            raise ValueError

    def mirrorAnnotations(self):
       if self.position == Position.left:
           self.position = Position.right
       elif self.position == Position.right:
           self.position = Position.left

    def mirrorPosesAndAnnotation(self):
        self.mirrorAnnotations()
        self.vec[0] *= -1

    def setKeypointID(self,id):
        self.position, self.feature = self.keypointIDToPositionFeaturePair(id)

    def getKeypointID(self) -> int:
        return self.PositionFeaturePairToKeypointID(self.position, self.feature)

    def is3D(self):
        return (self.vec.shape[0] == 3)

    @staticmethod
    def KeypointIDToPositionFeaturePair(id: int) -> (Position, Feature):
        position = id % 3
        feature = math.floor(id / 3)

        return Position(position), Feature(feature)

    @staticmethod
    def PositionFeaturePairToKeypointID(pos:Position, feat:Feature) -> int:
        return feat.value * 3 + pos.value
'''