import glob
import os
from configparser import ConfigParser
from functools import lru_cache
from typing import List, Dict
import re

import numpy as np

from common.camera import normalize_screen_coordinates
from common.h36m_dataset import Human36mDataset
from src.DataInterface.Camera import Camera
from src.DataInterface.DataInterface import DataInterface
from src.DataInterface.Human36mGT import Human36mGT
from src.DataInterface.MultiViewFrame import MultiViewFrame
from src.DataInterface.Sequence import Sequence, NpDimension
from src.Skeleton import CommonKeypointFormats
from src.Skeleton.CommonKeypointFormats import COMMON_KEYPOINT_FORMATS
from src.Skeleton.Skeleton import Skeleton
from src.h36m_noTears import Human36mNoTears
from src.Skeleton.Keypoint import Dimension as KpDim, Keypoint


class NoTears(DataInterface):

    def action_list(self, subject: str) -> List[str]:
        return self._dataPath[subject].keys()

    @property
    def subject_list(self) -> List[str]:
        return self._dataPath.keys()

    def __init__(self, config:ConfigParser, gtData:Human36mGT):

        self.config = config
        self._sequences = dict()
        self._accuracies = dict()
        #self.dataset = Human36mNoTears(config.get("NoTears", "2D"))
        self.human36mGT = gtData
        self.poseFiles = [f for f in glob.glob(os.path.join(config.get("NoTears", "2D"),"*.poses"))]
        self.poseFiles.sort()

        self._dataPath = dict()

        pattern = re.compile(r'^s_([0-9]*)_act_([0-9]*)_subact_([0-9]*)_ca_([0-9]*)_([0-9]*).poses$')
        self.fileread = re.compile(r'\[\s*([0-9.e+-]+)\s*,\s*([0-9.e+-]+)\s*,\s*([0-9.e+-]+)\s*\]')

        for f in self.poseFiles:
            fname = os.path.basename(f)
            m = pattern.match(fname)
            subject   = int(m.group(1))
            action_    = int(m.group(2))
            subaction = int(m.group(3))
            camera    = int(m.group(4))
            iterator  = int(m.group(5))
            subject = "S{}".format(subject)
            action = self.getActionFromIDs(subject,action_,subaction)

            if not subject in self._dataPath:
                self._dataPath[subject] = dict()

            if not action in self._dataPath[subject]:
                self._dataPath[subject][action] = dict()

            if not iterator in self._dataPath[subject][action]:
                self._dataPath[subject][action][iterator] = dict()

            self._dataPath[subject][action][iterator][camera] = f


    @lru_cache(maxsize=128)
    def get_sequence(self, subject:str, action:str):
        assert subject in self.subject_list, "Subject does not exist"
        assert action in self.action_list(subject)

        paths = self._dataPath[subject][action]

        camlist = list()
        for cam in self.human36mGT.get_sequence(subject, action).cameras:
            camlist.append(cam)

        _data = [None, None, None, None]
        # load Files
        for frameid,p in paths.items():
            for camidx,path in p.items():

                with open(path) as fd:
                    contstr = fd.read()

                    readdata = np.array(
                        [np.array([np.double(a), np.double(b), np.double(c)]) for a, b, c in self.fileread.findall(contstr)])
                    if readdata.shape[0] == 0:
                        print("No Prediction!")
                        readdata = np.ones([17, 3]) * np.nan

                    if readdata.shape[0] > 17:
                        rd = readdata.reshape(-1, 17, 3)
                        rd_argmax = np.argmax(np.sum(rd[:, :, 2], axis=1))
                        readdata = rd[rd_argmax]
                        print("WARNING: Using first Skeleton only")
                    if _data[camidx - 1] is None:
                        _data[camidx - 1] = np.empty((0, 17, 3))
                        #_data["accuracy"][camidx - 1] = np.empty((0, 17))

                    if readdata.shape[0] < 17:
                        print("No Prediction!")
                    readdata[..., :2] = normalize_screen_coordinates(readdata[..., :2], w=camlist[0].res[0],
                                                                               h=camlist[0].res[1])
                    #change y coordinates to bottom low, top high
                    #readdata[...,1] *= -1
                    readdata[...,2] = 1
                    #readdata *= np.array([[1,-1,1]])
                    _data[camidx - 1] = np.append(
                        _data[camidx - 1], readdata[None, :17, :], axis=0)

                    #_data["accuracy"][camidx - 1] = np.append(
                    #    _data["accuracy"][camidx - 1], readdata[None, :17, 2], axis=0)

        #build Sequence

        kps_ = np.array(_data)



        for kps in  _data:
            assert len(kps) == len( _data[0])

        seq = Sequence(camlist, kps_,
                       {NpDimension.CAM_ITER: [cam.id for cam in camlist],
                        NpDimension.FRAME_ITER: range(0,kps_.shape[1]*5,5),
                        NpDimension.KEYPOINT_ITER: COMMON_KEYPOINT_FORMATS.coco(),
                        NpDimension.KP_DATA: (KpDim.x, KpDim.y, KpDim.accuracy)}, name=subject+" "+action)
        return seq

    @property
    def sequence_list(self) -> Dict[str, Sequence]:
        return self._sequence_list

    def get_sequence_by_id(self, sequence_id: int) -> List[MultiViewFrame]:
        return self.get_sequence_by_name(self._sequence_list[sequence_id])

    def get_sequence_by_name(self, sequence_name: str) -> List[MultiViewFrame]:
        subject, action_name = self.sequence_name.split(".")
        cameras = Camera(cam= [c for c in self.dataset._cameras[subject]])
        frames = list()
        for pos in self.dataset._data[subject][action_name]["positions"]:
            frames.append(MultiViewFrame(self.generate_skeleton_list_from_positions(pos), None, cameras))

    def get_available_keypoints(self) -> List[Keypoint]:
        return COMMON_KEYPOINT_FORMATS.coco()

    @staticmethod
    def generate_skeleton_list_from_positions(positions):
        return [Skeleton(pos) for pos in positions]

    def genSkeleton(self, data) -> Skeleton:
        raise NotImplementedError

    @staticmethod
    @lru_cache(maxsize=None)
    def getActionFromIDs(subject, action, subaction):
        switcher = \
            {"S1": [
                ("_All 1", "_ALL"),
                ("Directions 1", "Directions"),
                ("Discussion 1", "Discussion"),
                ("Eating 2", "Eating"),
                ("Greeting 1", "Greeting"),
                ("Phoning 1", "Phoning"),
                ("Posing 1", "Posing"),
                ("Purchases 1", "Purchases"),
                ("Sitting 1", "Sitting 2"),
                ("SittingDown 2", "SittingDown"),
                ("Smoking 1", "Smoking"),
                ("Photo 1", "Photo"),  # ("TakingPhoto 1", "TakingPhoto"),
                ("Waiting 1", "Waiting"),
                ("Walking 1", "Walking"),
                ("WalkDog 1", "WalkDog"),  # ("WalkingDog 1", "WalkingDog"),
                ("WalkTogether 1", "WalkTogether")
            ],
                "S2": [
                    ("_All 2", "_ALL 1"),
                    ("Directions 1", "Directions"),
                    ("Discussion 1", "Discussion"),
                    ("Eating 1", "Eating 2"),
                    ("Greeting 1", "Greeting"),
                    ("Phoning 1", "Phoning"),
                    ("Posing 1", "Posing"),
                    ("Purchases 1", "Purchases"),
                    ("Sitting 1", "Sitting"),
                    ("SittingDown 2", "SittingDown 3"),
                    ("Smoking 1", "Smoking"),
                    ("Photo 1", "Photo"),  # ("TakingPhoto 1", "TakingPhoto"),
                    ("Waiting 1", "Waiting"),
                    ("Walking 1", "Walking"),
                    ("WalkDog 1", "WalkDog"),  # ("WalkingDog 1", "WalkingDog"),
                    ("WalkTogether 1", "WalkTogether")
                ],
                "S3": [
                    ("_All 1", "_ALL"),
                    ("Directions 1", "Directions"),
                    ("Discussion 1", "Discussion"),
                    ("Eating 1", "Eating 2"),
                    ("Greeting 1", "Greeting"),
                    ("Phoning 1", "Phoning"),
                    ("Posing 1", "Posing 2"),
                    ("Purchases 1", "Purchases"),
                    ("Sitting 1", "Sitting"),
                    ("SittingDown 1", "SittingDown"),
                    ("Smoking 1", "Smoking"),
                    ("Photo 1", "Photo"),  # ("TakingPhoto 1", "TakingPhoto"),
                    ("Waiting 1", "Waiting"),
                    ("Walking 1", "Walking 2"),
                    ("WalkDog 1", "WalkDog"),  # ("WalkingDog 1", "WalkingDog"),
                    ("WalkTogether 1", "WalkTogether")
                ],
                "S4": [
                    ("_All 1", "_ALL"),
                    ("Directions 1", "Directions"),
                    ("Discussion 1", "Discussion"),
                    ("Eating 1", "Eating"),
                    ("Greeting 1", "Greeting"),
                    ("Phoning 1", "Phoning"),
                    ("Posing 1", "Posing"),
                    ("Purchases 1", "Purchases"),
                    ("Sitting 1", "Sitting"),
                    ("SittingDown 1", "SittingDown 2"),
                    ("Smoking 1", "Smoking"),
                    ("Photo 1", "Photo"),  # ("TakingPhoto 1", "TakingPhoto"),
                    ("Waiting 1", "Waiting"),
                    ("Walking 1", "Walking"),
                    ("WalkDog 1", "WalkDog"),  # ("WalkingDog 1", "WalkingDog"),
                    ("WalkTogether 2", "WalkTogether 3")
                ],
                "S5": [
                    ("_All 1", "_ALL"),
                    ("Directions 1", "Directions 2"),
                    ("Discussion 2", "Discussion 3"),
                    ("Eating 1", "Eating"),
                    ("Greeting 1", "Greeting 2"),
                    ("Phoning 1", "Phoning"),
                    ("Posing 1", "Posing"),
                    ("Purchases 1", "Purchases"),
                    ("Sitting 1", "Sitting"),
                    ("SittingDown", "SittingDown 1"),
                    ("Smoking 1", "Smoking"),
                    ("Photo", "Photo 2"),  # ("TakingPhoto 1", "TakingPhoto"),
                    ("Waiting 1", "Waiting 2"),
                    ("Walking 1", "Walking"),
                    ("WalkDog 1", "WalkDog"),  # ("WalkingDog 1", "WalkingDog"),
                    ("WalkTogether 1", "WalkTogether")
                ],
                "S6": [
                    ("_All 1", "_ALL"),
                    ("Directions 1", "Directions"),
                    ("Discussion 1", "Discussion"),
                    ("Eating 1", "Eating 2"),
                    ("Greeting 1", "Greeting"),
                    ("Phoning 1", "Phoning"),
                    ("Posing 2", "Posing"),
                    ("Purchases 1", "Purchases"),
                    ("Sitting 1", "Sitting 2"),
                    ("SittingDown 1", "SittingDown"),
                    ("Smoking 1", "Smoking"),
                    ("Photo", "Photo 1"),  # ("TakingPhoto 1", "TakingPhoto"),
                    ("Waiting 3", "Waiting"),
                    ("Walking 1", "Walking"),
                    ("WalkDog 1", "WalkDog"),  # ("WalkingDog 1", "WalkingDog"),
                    ("WalkTogether 1", "WalkTogether")
                ],
                "S7": [
                    ("_All 1", "_ALL"),
                    ("Directions 1", "Directions"),
                    ("Discussion 1", "Discussion"),
                    ("Eating 1", "Eating"),
                    ("Greeting 1", "Greeting"),
                    ("Phoning 2", "Phoning"),
                    ("Posing 1", "Posing"),
                    ("Purchases 1", "Purchases"),
                    ("Sitting 1", "Sitting"),
                    ("SittingDown", "SittingDown 1"),
                    ("Smoking 1", "Smoking"),
                    ("Photo", "Photo 1"),  # ("TakingPhoto 1", "TakingPhoto"),
                    ("Waiting 1", "Waiting 2"),
                    ("Walking 1", "Walking 2"),
                    ("WalkDog 1", "WalkDog"),  # ("WalkingDog 1", "WalkingDog"),
                    ("WalkTogether 1", "WalkTogether")
                ],
                "S8": [
                    ("_All 1", "_ALL"),
                    ("Directions 1", "Directions"),
                    ("Discussion 1", "Discussion"),
                    ("Eating 1", "Eating"),
                    ("Greeting 1", "Greeting"),
                    ("Phoning 1", "Phoning"),
                    ("Posing 1", "Posing"),
                    ("Purchases 1", "Purchases"),
                    ("Sitting 1", "Sitting"),
                    ("SittingDown", "SittingDown 1"),
                    ("Smoking 1", "Smoking"),
                    ("Photo 1", "Photo"),  # ("TakingPhoto 1", "TakingPhoto"),
                    ("Waiting 1", "Waiting"),
                    ("Walking 1", "Walking"),
                    ("WalkDog 1", "WalkDog"),  # ("WalkingDog 1", "WalkingDog"),
                    ("WalkTogether 1", "WalkTogether 2")
                ],
                "S9": [
                    ("_All 1", "_ALL"),
                    ("Directions 1", "Directions"),
                    ("Discussion 1", "Discussion 2"),
                    ("Eating 1", "Eating"),
                    ("Greeting 1", "Greeting"),
                    ("Phoning 1", "Phoning"),
                    ("Posing 1", "Posing"),
                    ("Purchases 1", "Purchases"),
                    ("Sitting 1", "Sitting"),
                    ("SittingDown", "SittingDown 1"),
                    ("Smoking 1", "Smoking"),
                    ("Photo 1", "Photo"),  # ("TakingPhoto 1", "TakingPhoto"),
                    ("Waiting 1", "Waiting"),
                    ("Walking 1", "Walking"),
                    ("WalkDog 1", "WalkDog"),  # ("WalkingDog 1", "WalkingDog"),
                    ("WalkTogether 1", "WalkTogether")
                ],
                "S10": [
                    ("_All 2", "_ALL 1"),
                    ("Directions 1", "Directions"),
                    ("Discussion 1", "Discussion 2"),
                    ("Eating 1", "Eating"),
                    ("Greeting 1", "Greeting"),
                    ("Phoning 1", "Phoning"),
                    ("Posing 1", "Posing"),
                    ("Purchases 1", "Purchases"),
                    ("Sitting 1", "Sitting"),
                    ("SittingDown", "SittingDown 1"),
                    ("Smoking 2", "Smoking"),
                    ("Photo 1", "Photo"),  # ("TakingPhoto 1", "TakingPhoto"),
                    ("Waiting 1", "Waiting"),
                    ("Walking 1", "Walking"),
                    ("WalkDog 1", "WalkDog"),  # ("WalkingDog 1", "WalkingDog"),
                    ("WalkTogether 1", "WalkTogether")
                ],
                "S11": [
                    ("_All 1", "_ALL"),
                    ("Directions 1", "Directions"),
                    ("Discussion 1", "Discussion 2"),
                    ("Eating 1", "Eating"),
                    ("Greeting 2", "Greeting"),
                    ("Phoning 3", "Phoning 2"),
                    ("Posing 1", "Posing"),
                    ("Purchases 1", "Purchases"),
                    ("Sitting 1", "Sitting"),
                    ("SittingDown", "SittingDown 1"),
                    ("Smoking 2", "Smoking"),
                    ("Photo 1", "Photo"),  # ("TakingPhoto 1", "TakingPhoto"),
                    ("Waiting 1", "Waiting"),
                    ("Walking 1", "Walking"),
                    ("WalkDog 1", "WalkDog"),  # ("WalkingDog 1", "WalkingDog"),
                    ("WalkTogether 1", "WalkTogether")
                ]
            }
        return switcher[subject][action - 1][subaction - 1]