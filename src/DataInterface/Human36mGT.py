from configparser import ConfigParser
from functools import lru_cache
from typing import List, Dict

from common.h36m_dataset import Human36mDataset
from src.DataInterface.Camera import Camera
from src.DataInterface.DataInterface import DataInterface
from src.DataInterface.MultiViewFrame import MultiViewFrame
from src.DataInterface.Sequence import Sequence, NpDimension
from src.Skeleton.CommonKeypointFormats import COMMON_KEYPOINT_FORMATS
from src.Skeleton.Keypoint import Dimension as KpDim, Keypoint, Position, Feature


class Human36mGT(DataInterface):
    @property
    def subject_list(self) -> List[str]:
        return list(self.dataset.subjects())

    def action_list(self, subject:str) -> List[str]:
        if subject in self.dataset._data:
            return list(self.dataset._data[subject].keys())
        return list()

    def __init__(self, config:ConfigParser):
        self.config = config
        self.dataset = Human36mDataset(config.get("Human3.6m", "3D"))

    @property
    def sequence_list(self) -> Dict[str, Sequence]:
        raise NotImplementedError

    #@lru_cache(maxsize=10)
    def get_sequence(self, subject: str, action: str) -> Sequence:
        if not subject in self.subject_list:
            raise IndexError("subject \"" + subject +"\" not in subject list")
        if not action in self.action_list(subject):
            raise IndexError("no action \""+ action + "\" for subject \""+subject+"\"")
        cams = list()

        for cam in self.dataset.cameras()[subject]:
            cams.append(Camera(cam))

        #for kps in self.dataset[subject][action]["positions"]:
        #    assert len(kps) == len(self.dataset[subject][action]["positions"][0])
        nparray = self.dataset[subject][action]["positions"]
        seq = Sequence(cams, nparray,
                 {NpDimension.FRAME_ITER: range(nparray.shape[0]),
                  NpDimension.KEYPOINT_ITER: COMMON_KEYPOINT_FORMATS.unknownWithMHipAndTopHead(),
                  NpDimension.KP_DATA: (KpDim.x,KpDim.y,KpDim.z)}, name=subject+" "+action)
        return seq

    def get_available_keypoints(self) -> List[Keypoint]:
        return COMMON_KEYPOINT_FORMATS.unknownWithMHipAndTopHead()

