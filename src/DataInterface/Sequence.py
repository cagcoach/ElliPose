import traceback
from collections import OrderedDict
from enum import Enum
from typing import List, Dict

import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import scipy

from src.DataInterface.Camera import Camera
from src.DataInterface.MultiViewFrame import MultiViewFrame
from src.Skeleton.Keypoint import Keypoint, Feature, Position
from src.Skeleton.Keypoint import Dimension as KpDim


class NpDimension(Enum):
    CAM_ITER = 0
    FRAME_ITER = 1
    KEYPOINT_ITER = 2
    KP_DATA = 3
    RGB_DATA = 4
    SKELETON = 5

class SequenceFormat:
    def __init__(self, dimensions: List[NpDimension], cams: List[str], keypoints: List[Keypoint]):
        pass

class Sequence:

    def __init__(self, cameras:List[Camera], nparray:np.array, npdimensions:OrderedDict, name:str = None):
        '''
        cameras: List of Camera
        nparray: the actual data as numpy array. Dimensions and orderings, can be arbitrary but they have to be declared in npdimensions
        npdimensions: OrderedDict of Lists. Keys in the Ordered Dict are of enum-type NpDimensions. The values are list containing the definition of what the data actual corresponds to.
            Make sure that len(nparray.shape) == len(npdimensions) and nparray.shape[i] == len(npdimensions[i-th element].value)
            - FRAME_ITER, CAM_ITER, SKELETON: might be an int value or string, corresponding giving a hint to a frame/camera/skeleton name or id
            - KEYPOINT_ITER: is of type Keypoint
            - KP_DATA: is of type Keypoint.Dimension
        '''
        npdimensions = OrderedDict(npdimensions)
        for num, dim in enumerate(npdimensions.items()):
            if dim[0] == NpDimension.CAM_ITER:
                assert nparray.shape[num] == len(cameras)
                break

        self._cameras = cameras
        self.npdimensions = npdimensions
        self.nparray = nparray
        self.name = name

    def __len__(self):
        for num,dim in enumerate(self.npdimensions.items()):
            if dim.key == NpDimension.FRAME_ITER:
                return self.nparray.shape[num]

    def get(self, format:OrderedDict = None):
        '''
        :param format: OrderedDict of Lists, Single Value or None. Keys in the Ordered Dict are of enum-type NpDimensions. The values are list or values containing the definition of what the data actual corresponds to, values which do not correspond to any value, are filled with np.nan.
            When single values are provided, the output dimension is collapsed. If the value is None, the data order is left untouched.

            Make sure that len(nparray.shape) == len(npdimensions) and nparray.shape[i] == len(npdimensions[i-th element].value)
            - FRAME_ITER, CAM_ITER, SKELETON: might be an int value or string, corresponding giving a hint to a frame/camera/skeleton name or id
            - KEYPOINT_ITER: is of type Keypoint
            - KP_DATA: is of type Keypoint.Dimension
        :return:
        '''
        if format is None:
            return self.nparray, self.npdimensions

        format = OrderedDict(format).copy()
        nparr, newform = Sequence.convertToFormat(self.nparray,self.npdimensions,format.keys())

        tup = list()
        for k,v in format.copy().items():
            if type(v) == property:
                v = v.fget()

            if v is None:
                tup.append(slice(None))
                format[k] = self.npdimensions[k]
            elif type(v) == list or type(v) == tuple or type(v) == range:
                try:
                    tup.append(tuple(newform[k].index(v_) for v_ in v))
                except ValueError as e: # Fill missing columns with nan
                    nparr = np.insert(nparr,0,np.nan,axis=len(tup))
                    l = list()
                    for v_ in v:
                        if v_ in newform[k]:
                            l.append(newform[k].index(v_) + 1)
                        else:
                            l.append(0)
                    tup.append(tuple(l))
                    formatk = []
                    for i in range(len(l)):
                        if l[i] == 0:
                            formatk.append(None)
                        else:
                            formatk.append(format[k][i])
                    format[k] = formatk
            else:
                tup.append(newform[k].index(v))
                format.pop(k)

        for i in range(len(tup), 0, -1):
            nparr = nparr[tuple(tup[i-1] if j == i-1 else slice(None) for j in range(i))]

        return nparr, format


    def addCamera(self, camera:Camera):
        assert (len(self._frames == 0)), "Cameras can only be added while no frames are set"
        self._cameras[camera.id] = camera

    def visualizeSingleFrame(self, selection:dict, additionalPoints:np.array=None):
        '''Example:  self.visualizeSingleFrame({NpDimension.FRAME_ITER: 50,
                                 NpDimension.KEYPOINT_ITER: None,
                                 NpDimension.KP_DATA: (KpDim.x, KpDim.y, KpDim.z)})'''
        if additionalPoints is None:
            additionalPoints = np.empty((0,3))
        formatDict = OrderedDict()
        for k,v in selection.items():
            if k != NpDimension.KP_DATA:
                formatDict[k] = v
        if NpDimension.KP_DATA in selection:
            formatDict[NpDimension.KP_DATA] = selection[NpDimension.KP_DATA]
        else:
            if KpDim.z in self.npdimensions[NpDimension.KP_DATA]:
                formatDict[NpDimension.KP_DATA] = (KpDim.x, KpDim.y, KpDim.z)
            else:
                formatDict[NpDimension.KP_DATA] = (KpDim.x, KpDim.y)
        data, format = self.get(formatDict)

        datar = data.reshape(-1,data.shape[-1])
        matplotlib.use("TKAgg")
        fig = plt.figure(self.name)
        if data.shape[-1] == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim3d([-2.5, 2.5])
            x = datar[:, 0]
            ax.set_ylim3d([-2.5, 2.5])
            y = datar[:, 1]
            ax.set_zlim3d([-0.1, 4.9])
            z = datar[:, 2]
            _, = ax.plot(x, y, z, linestyle="", marker="o", color="blue")
            _, = ax.plot(additionalPoints[:,0],additionalPoints[:,1],additionalPoints[:,2], linestyle="", marker="o", color="yellow")

        else:
            ax = fig.add_subplot(111)
            ax.set_xlim([-1.5, 1.5])
            x = datar[:, 0]
            ax.set_ylim([-1.5, 1.5])
            y = datar[:, 1]
            _, = ax.plot(x, y, linestyle="", marker="o", color="blue")
            _, = ax.plot(additionalPoints[:, 0], additionalPoints[:, 1], linestyle="",
                         marker="o", color="yellow")


        plt.show(block=False)
        fig.canvas.draw()

    def interpolateCenterFromLeftRight(self, feature:Feature) -> bool:
        if Keypoint(Position.center, feature) in self.npdimensions[NpDimension.KEYPOINT_ITER]:
            return False
        if not Keypoint(Position.left, feature) in self.npdimensions[NpDimension.KEYPOINT_ITER]:
            return False
        if not Keypoint(Position.right, feature) in self.npdimensions[NpDimension.KEYPOINT_ITER]:
            return False
        dimindex = list(self.npdimensions.keys()).index(NpDimension.KEYPOINT_ITER)
        leftindex = self.npdimensions[NpDimension.KEYPOINT_ITER].index(Keypoint(Position.left,feature))
        rightindex = self.npdimensions[NpDimension.KEYPOINT_ITER].index(Keypoint(Position.right, feature))
        self.npdimensions[NpDimension.KEYPOINT_ITER].append(Keypoint(Position.center,feature))
        swax = self.nparray.swapaxes(0,dimindex)
        swax = np.concatenate([swax, (swax[(leftindex,),]+ swax[(rightindex,),])/2.])
        self.nparray = swax.swapaxes(0,dimindex)
        return True


    def triangulate3DPoints(self, cam1, cam2, blur=False, useExtrinsics = True, minaccuracy = 0.8, topleftorigin = False) -> ("Sequence", np.ndarray, np.ndarray):
        kp, f1 = self.get(OrderedDict({NpDimension.CAM_ITER: [cam1,cam2],
                                            NpDimension.FRAME_ITER: None,
                                            NpDimension.KEYPOINT_ITER: None,
                                            NpDimension.KP_DATA: (KpDim.x, KpDim.y)}))


        cams = (self.cameras[self.cameras.index(cam1)].copy(),self.cameras[self.cameras.index(cam2)].copy())
        #realCameraDist = np.linalg.norm(cams[0].translation - cams[1].translation)
        acc, fa1 = self.get(OrderedDict({NpDimension.CAM_ITER: [cam1,cam2],
                                              NpDimension.FRAME_ITER: None,
                                              NpDimension.KEYPOINT_ITER: None,
                                              NpDimension.KP_DATA: KpDim.accuracy}))

        accprod = acc.prod(axis=0)

        if blur:
            kp = scipy.ndimage.gaussian_filter1d(kp, 1.5, axis=1)
        if topleftorigin:
            kp[:,:,1] *= -1
            cams[0].center[1] *= -1
            cams[1].center[1] *= -1
        kp = kp.reshape(2,-1, 2)
        acc_ = accprod.reshape(-1) >= minaccuracy
        goodPredictionMask = acc_

        if useExtrinsics:
            raise NotImplementedError

        for c in cams:
            c.translation = None
            c.orientation = None
        try:
            E = cv2.findEssentialMat(kp[0][goodPredictionMask].copy(), kp[1][goodPredictionMask].copy(),
                                     cameraMatrix1=cams[0].intrinsicMatrix, cameraMatrix2=cams[1].intrinsicMatrix,
                                     distCoeffs1=None,
                                     distCoeffs2=None, method=cv2.RANSAC, prob=0.9999, threshold=0.001)
        except cv2.error:
            traceback.print_exc()
            print("cverror")

        kp1__ = (kp[0][goodPredictionMask] - cams[0].intrinsicMatrix[None, :2, 2]) / cams[0].intrinsicMatrix.diagonal()[None, :2]
        kp2__ = (kp[1][goodPredictionMask] - cams[1].intrinsicMatrix[None, :2, 2]) / cams[1].intrinsicMatrix.diagonal()[None, :2]

        retval, R, t, mask, _ = cv2.recoverPose(E[0], kp1__, kp2__, cameraMatrix=np.eye(3), distanceThresh=1000);
        scale = 1 / np.linalg.norm(t)
        if scale == np.nan:
            raise ValueError()
        t *= scale

        cams[0].rotationMatrix = np.eye(3)
        cams[0].translation = np.zeros(3)
        cams[1].rotationMatrix = R
        cams[1].translation = t.squeeze()

        P = np.append(cams[0].intrinsicMatrix, np.array([[0], [0], [0]]), axis=1)
        P_ = cams[1].intrinsicMatrix @ np.append(R, t, axis=1)

        triangulatedPoints = cv2.triangulatePoints(P, P_, kp[0].transpose(), kp[1].transpose())
        triangulatedPoints /= triangulatedPoints[3, :]
        triangulatedPoints = triangulatedPoints.transpose()[:, :3].reshape(-1, 17, 3)

        nparraynpr = np.stack([triangulatedPoints[..., 0], triangulatedPoints[..., 1], triangulatedPoints[..., 2], accprod], axis=2)
        return Sequence([cams[0],cams[1]],nparraynpr,npdimensions= OrderedDict([
            (NpDimension.FRAME_ITER, f1[NpDimension.FRAME_ITER]),
            (NpDimension.KEYPOINT_ITER, f1[NpDimension.KEYPOINT_ITER]),
            (NpDimension.KP_DATA, (KpDim.x, KpDim.y, KpDim.z, KpDim.accuracy))
        ]), name="triangulated "+self.name), P, P_

    @property
    def frames(self):
        return self._frames

    @property
    def cameras(self):
        return self._cameras

    @staticmethod
    def convertToFormat(orignparray, orignpdimensions, format=List[NpDimension]):
        assert len(format) == len(orignpdimensions)
        src = list()
        dst = list()

        for dst_ in range(len(format)):
            src_ = list(orignpdimensions.keys()).index(list(format)[dst_])
            src.append(src_)
            dst.append(dst_)
        outnparray = np.moveaxis(orignparray, src, dst)
        new_npdimensions = OrderedDict({k: orignpdimensions[k] for k in format})
        outnpdimensions = new_npdimensions
        return outnparray, outnpdimensions