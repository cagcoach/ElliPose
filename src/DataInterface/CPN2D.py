from functools import lru_cache
from typing import List, Dict

from common.camera import normalize_screen_coordinates
from src.DataInterface.DataInterface import DataInterface
from src.DataInterface.Human36mGT import Human36mGT
from src.DataInterface.MultiViewFrame import MultiViewFrame
from src.DataInterface.Sequence import Sequence, NpDimension
from src.Skeleton.Keypoint import Dimension as KpDim

import numpy as np

from src.Skeleton.CommonKeypointFormats import COMMON_KEYPOINT_FORMATS


class CPN2D(DataInterface):
    def __init__(self, config: dict, gtData:Human36mGT):
        self.config = config
        self._sequences = dict()
        self._accuracies = dict()
        # self.dataset = Human36mNoTears(config.get("NoTears", "2D"))
        self.human36mGT = gtData
        self.dataset = dict(np.load(config.get("CPN", "2D"), allow_pickle=True))

        self.keypoints = self.dataset['positions_2d'].item()

    @property
    def sequence_list(self) -> Dict[str, Sequence]:
        raise NotImplementedError

    @property
    def subject_list(self) -> List[str]:
        return list(self.keypoints.keys())

    def action_list(self, subject: str) -> List[str]:
        return list(self.keypoints[subject].keys())

    @lru_cache(maxsize=128)
    def get_sequence(self, subject: str, action: str) -> Sequence:
        assert subject in self.subject_list, "Subject does not exist"
        assert action in self.action_list(subject)

        camlist = list()
        for cam in self.human36mGT.get_sequence(subject, action).cameras:
            camlist.append(cam)


        kpslen = min([len(i) for i in self.keypoints[subject][action]])
        nparr = np.empty((len(camlist),kpslen,17,3))
        for cam_idx, kps in enumerate(self.keypoints[subject][action]):
            cam = camlist[cam_idx]

            kps = normalize_screen_coordinates(kps[..., :2], w=cam.res[0], h=cam.res[1])
            nparr[cam_idx] = np.stack([kps[:kpslen, :,0],kps[:kpslen, :,1],np.ones([kpslen,17])],axis=2)


        seq = Sequence(camlist, nparr,
                       {NpDimension.CAM_ITER: [cam.id for cam in camlist],
                        NpDimension.FRAME_ITER: range(0, nparr.shape[1]),
                        NpDimension.KEYPOINT_ITER: COMMON_KEYPOINT_FORMATS.unknownWithMHipAndTopHead(),
                        NpDimension.KP_DATA: (KpDim.x, KpDim.y, KpDim.accuracy)}, name=subject+" "+action)
        return seq

    def get_available_keypoints(self) -> List[int]:
        return COMMON_KEYPOINT_FORMATS.unknownWithMHipAndTopHead()