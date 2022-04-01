from abc import abstractmethod, abstractproperty
from functools import lru_cache
from typing import List, Dict

from src.DataInterface.MultiViewFrame import MultiViewFrame

from src.DataInterface.Sequence import Sequence
from src.Skeleton.Keypoint import Keypoint


class DataInterface:
    @abstractmethod
    def __init__(self, config: dict):
        raise NotImplementedError

    @property
    @abstractmethod
    def sequence_list(self) -> Dict[str, Sequence]:
        raise NotImplementedError

    @property
    @abstractmethod
    def subject_list(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def action_list(self, subject:str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    @lru_cache
    def get_sequence(self, subject: str, action: str) -> Sequence:
        raise NotImplementedError

    @abstractmethod
    def get_available_keypoints(self) -> List[Keypoint]:
        raise NotImplementedError