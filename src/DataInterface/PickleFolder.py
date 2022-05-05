import glob
import os
import pickle
import re
from functools import lru_cache
from typing import List, Dict
import numpy as np

from src.DataInterface.DataInterface import DataInterface
from src.DataInterface.MultiViewFrame import MultiViewFrame
from src.DataInterface.Sequence import Sequence, NpDimension
from src.Skeleton.Keypoint import Keypoint


class PickleFolder(DataInterface):
    def __init__(self, config: dict, datakey: None):
        pattern = re.compile(r'^(S[0-9]*)_(.*).pkl$')
        self.pickleFiles = [f for f in glob.glob(os.path.join(config.get("exec", "outpath"), "*.pkl"))]
        self._dataPath = dict()
        self.datakey = datakey

        for f in self.pickleFiles:
            fname = os.path.basename(f)
            m = pattern.match(fname)
            subject   = m.group(1)
            action    = m.group(2)

            if not subject in self._dataPath:
                self._dataPath[subject] = dict()

            if not action in self._dataPath[subject]:
                self._dataPath[subject][action] = f

    @property
    def sequence_list(self) -> Dict[str, Sequence]:
        raise NotImplementedError

    @property
    def subject_list(self) -> List[str]:
        return list(self._dataPath.keys())

    def action_list(self, subject: str) -> List[str]:
        return list(self._dataPath[subject].keys())

    #@lru_cache(maxsize=10)
    def get_sequence(self, subject: str, action: str) -> Sequence:
        with open(self._dataPath[subject][action], "rb") as input_file:
            datadict = pickle.load(input_file)
        if self.datakey is None:
            return datadict
        else:
            return datadict[self.datakey]

    def get_available_keypoints(self) -> List[Keypoint]:
        subject = self.subject_list[0]
        action = self.action_list(subject)[0]
        return self.get_sequence(subject, action).npdimensions[NpDimension.KEYPOINT_ITER]

    def get_bestA_for_Sequence(self, subject: str, action: str):
        with open(self._dataPath[subject][action], "rb") as input_file:
            datadict = pickle.load(input_file)
        return datadict["bestA"] if "bestA" in datadict else np.nan;