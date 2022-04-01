import numpy as np
from scipy.spatial.transform import Rotation


class Camera:
    center = None
    res = None
    focal_length = None
    radial_distortion = None
    tangential_distortion = None
    azimuth = None
    translation = None
    orientation = None

    def __init__(self, cam: dict):
        if type(cam) == dict:
            self.center = cam['center']
            self.focal_length = cam['focal_length']
            self.res = np.array([cam["res_w"],cam["res_h"]])
            self.radial_distortion = cam['radial_distortion']
            self.tangential_distortion = cam["tangential_distortion"]
            self.azimuth = cam["azimuth"]
            self.translation = cam["translation"]
            self.orientation = cam["orientation"]
            self.id = cam["id"]
        elif type(cam) == Camera:
            self.center = cam.center
            self.focal_length = cam.focal_length
            self.res = cam.res.copy()
            self.radial_distortion = cam.radial_distortion
            self.tangential_distortion = cam.tangential_distortion
            self.azimuth = cam.azimuth
            self.translation = cam.translation
            self.orientation = cam.orientation
            self.id = cam.id

    def __eq__(self, other):
        if type(other) == int or type(other) == str:
            return self.id == other

        return self.id == other.id \
           and self.center == other.center \
           and self.focal_length == other.focal_length \
           and self.res == other.res \
           and self.radial_distortion ==other.radial_distortion \
           and self.tangential_distortion == other.tangential_distortion \
           and self.azimuth == other.azimuth \
           and self.translation == other.translation \
           and self.orientation == other.orientation \



    def getIntrinsics(self):
        return np.concatenate((self.focal_length, #2
                               self.center, #2
                               self.radial_distortion, #3
                               self.tangential_distortion)) #2

    @property
    def intrinsic(self):
        return self.getIntrinsics()

    def normalize_screen_coordinates(self):
        self.normalize_screen_coordinates(self.center)

    def normalize_screen_coordinates(self, X):
        assert X.shape[-1] == 2

        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return X / self.res[0] * 2 - [1, self.res[1] / self.res[0]]

    @property
    def rotationMatrix(self):
        return Rotation.from_quat(self.orientation[(1, 2, 3, 0),]).as_matrix()

    @rotationMatrix.setter
    def rotationMatrix(self, value):
        self.orientation = Rotation.from_matrix(value).as_quat()[(3,0,1,2),]

    @property
    def intrinsicMatrix(self):
        return np.array([[self.focal_length[0], 0, self.center[0]],
                  [0, self.focal_length[1], self.center[1]],
                  [0, 0, 1]])

    def __copy__(self):
        return Camera(self)

    def copy(self):
        return Camera(self)