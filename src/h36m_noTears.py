import re

from external.VideoPose3D.common.camera import normalize_screen_coordinates
from external.VideoPose3D.common.h36m_dataset import h36m_skeleton, h36m_cameras_extrinsic_params, h36m_cameras_intrinsic_params
from external.VideoPose3D.common.mocap_dataset import MocapDataset
import copy
import numpy as np
import glob
import os.path



class Human36mNoTears(MocapDataset):

    @staticmethod
    def getActionFromIDs(subject,action,subaction):
        switcher = \
            {"S1": [
                 ("_All 1","_ALL"),
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
                 ("Photo 1","Photo"), #("TakingPhoto 1", "TakingPhoto"),
                 ("Waiting 1", "Waiting"),
                 ("Walking 1", "Walking"),
                 ("WalkDog 1", "WalkDog"), #("WalkingDog 1", "WalkingDog"),
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
        return switcher[subject][action-1][subaction-1]

    def __init__(self, path, skeleton=h36m_skeleton):
        super().__init__(fps=50, skeleton=skeleton)
        self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                cam.update(h36m_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')

                # Normalize camera frame
                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'],
                                                             h=cam['res_h']).astype('float32')
                cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2
                if 'translation' in cam:
                    cam['translation'] = cam['translation'] / 1000  # mm to meters

                # Add intrinsic parameters vector
                cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                                   cam['center'],
                                                   cam['radial_distortion'],
                                                   cam['tangential_distortion']))

        pattern = re.compile(r'^s_([0-9]*)_act_([0-9]*)_subact_([0-9]*)_ca_([0-9]*)_([0-9]*).poses$')
        fileread = re.compile(r'\[\s*([0-9.e+-]+)\s*,\s*([0-9.e+-]+)\s*,\s*([0-9.e+-]+)\s*\]')
        poseFiles = [f for f in glob.glob(os.path.join(path,"*.poses"))]
        poseFiles.sort()
        self._data = dict()
        for f in poseFiles:
            fname = os.path.basename(f)
            m = pattern.match(fname)
            subject   = int(m.group(1))
            action    = int(m.group(2))
            subaction = int(m.group(3))
            camera    = int(m.group(4))
            iterator  = int(m.group(5))
            subject = "S{}".format(subject)
            action = self.getActionFromIDs(subject,action,subaction)

            if not subject in self._data:
                self._data[subject] = dict()

            if not action in self._data[subject]:
                self._data[subject][action] = {
                    "positions": [None,None,None,None],
                    "accuracy": [None,None,None,None],
                    "cameras": self._cameras[subject]
                }

            with open(f) as fd:
                contstr = fd.read()

                readdata = np.array([np.array([np.double(a),np.double(b),np.double(c)]) for a,b,c in fileread.findall(contstr)])
                if readdata.shape[0] == 0:
                    print("No Prediction!")
                    readdata = np.ones([17,3]) * np.nan

                if readdata.shape[0] > 17:
                    rd = readdata.reshape(-1, 17, 3)
                    rd_argmax = np.argmax(np.sum(rd[:,:,2],axis=1))
                    readdata = rd[rd_argmax]
                    print("WARNING: Using first Skeleton only")
                if self._data[subject][action]["positions"][camera-1] is None:
                    self._data[subject][action]["positions"][camera - 1] = np.empty((0,17,2))
                    self._data[subject][action]["accuracy"][camera - 1] = np.empty((0, 17))

                if readdata.shape[0] < 17:
                    print("No Prediction!")
                self._data[subject][action]["positions"][camera-1] = np.append(self._data[subject][action]["positions"][camera-1],readdata[None,:17,:2], axis=0)
                self._data[subject][action]["accuracy"][camera-1] = np.append(self._data[subject][action]["accuracy"][camera-1],readdata[None,:17,2],axis=0)

    def supports_semi_supervised(self):
        return True
