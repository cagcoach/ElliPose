from typing import List

from src.DataInterface.Camera import Camera
from src.Skeleton.Skeleton import Skeleton


class MultiViewFrame:

    def __init__(self, poses : List[Skeleton], images : list, cameras : List[Camera], names : list):
        self._numberOfDirections = len(poses)
        assert (images is None or len(images) == self._numberOfDirections)
        assert (cameras is None or len(images) == self._numberOfDirections)
        assert (names is None or len(images) == self._numberOfDirections)
        self._poses = poses
        self._images = images
        self._cameras = cameras
        self._namedict = {value:counter for counter, value in enumerate(names)}


    def __len__(self):
        return self._numberOfDirections

    @property
    def cameras(self):
        return self._cameras


    #@cameras.__getattribute__
    #def _cameras_getattr(self, name) -> Camera:
    #    if type(name) == str:
    #        return self._cameras[self._namedict[name]];
    #    elif type(name) == int:
    #        return self._cameras[name]

    @property
    def images(self):
        return self._images

    #@images.__getattribute__
    #def _images_getattr(self, name):
    #    if type(name) == str:
    #        return self._images[self._namedict[name]];
    #    elif type(name) == int:
    #        return self._images[name]

    @property
    def poses(self):
        return self._poses

    #@poses.__getattribute__
    #def _poses_getattr(self, name) -> Skeleton:
    #    if type(name) == str:
    #        return self._poses[self._namedict[name]];
    #    elif type(name) == int:
    #        return self._poses[name]
