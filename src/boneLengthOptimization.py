import traceback
from collections import OrderedDict
#from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
import cvxpy as cp
import random

from src.DataInterface.Camera import Camera
from src.DataInterface.Sequence import NpDimension,Sequence
from src.Skeleton.Bones import Bones
from src.Skeleton.Keypoint import Dimension as KpDim


class boneLengthOptimization():
    @staticmethod
    def globalDist(x, x_):
        return (x[0] - x_[0]) ** 2 + (x[1] - x_[1]) ** 2 + (x[2] - x_[2]) ** 2

    @staticmethod
    def projdist(p1, p_1, x_, P, P_):
        p2 = P[:, :3] @ x_ + P[:, 3]
        p2 *= p1[2]  # p1[2] is already weighted by acc1 thus no weight needed

        p_2 = P_[:, :3] @ x_ + P_[:, 3]
        p_2 *= p_1[2]  # p_1[2] is already weighted by acc2 thus no weight needed

        # return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p_1[0]-p_2[0])**2 + (p_1[1]-p_2[1])**2
        return (p1[0] - p2[0]) ** 2 \
               + (p1[1] - p2[1]) ** 2 \
               + (p_1[0] - p_2[0]) ** 2 \
               + (p_1[1] - p_2[1]) ** 2

    @staticmethod
    def bonediff(bone, x, tbij):
        xbj0 = x[bone[0].value]
        xbj1 = x[bone[1].value]
        return ((xbj0[0] - xbj1[0] - tbij[0]) ** 2
                + (xbj0[1] - xbj1[1] - tbij[1]) ** 2
                + (xbj0[2] - xbj1[2] - tbij[2]) ** 2)

    @staticmethod
    def bonelength(poses, bonemat=None):

        if bonemat is None:
            raise ValueError("Bonemat is None")
        poses_ = poses.copy()
        nanvalues = np.array(np.where(np.isnan(poses_))).transpose()

        for numb in range(len(nanvalues)):
            i = nanvalues[numb]
            poses_[i[0],i[1],i[2]] = numb * 1e20

        dimensiondiff = (np.moveaxis(poses_, 1, 2).reshape(-1,17) @ bonemat).reshape(-1,3,bonemat.shape[1])
        dimensiondiff = np.moveaxis(dimensiondiff, 1,2)

        dimensiondiff[dimensiondiff<=-9e19] = np.nan
        dimensiondiff[dimensiondiff>=9e19] = np.nan

        length = np.sqrt(np.sum(np.square(dimensiondiff), axis=2))

        return length, bonemat

    @staticmethod
    def optimize3DPoints(P: np.array, P_:np.array, sequence2d: Sequence, sequence3d: Sequence, cam1: Camera, cam2: Camera):
        prediction, f = sequence3d.get(format=OrderedDict([(NpDimension.FRAME_ITER, None),
                                                  (NpDimension.KEYPOINT_ITER, None),
                                                  (NpDimension.KP_DATA, (KpDim.x, KpDim.y, KpDim.z))]))

        prediction_acc, f_acc = sequence3d.get(format=OrderedDict([(NpDimension.FRAME_ITER, f[NpDimension.FRAME_ITER]),
                                                  (NpDimension.KEYPOINT_ITER, f[NpDimension.KEYPOINT_ITER]),
                                                  (NpDimension.KP_DATA, KpDim.accuracy)]))

        x, _ = sequence2d.get(format=OrderedDict([(NpDimension.CAM_ITER, [cam1, cam2]),
                                       (NpDimension.FRAME_ITER, f[NpDimension.FRAME_ITER]),
                                       (NpDimension.KEYPOINT_ITER, f[NpDimension.KEYPOINT_ITER]),
                                       (NpDimension.KP_DATA, (KpDim.x, KpDim.y))]))
        x_acc, _ = sequence2d.get(format=OrderedDict([(NpDimension.CAM_ITER, [cam1, cam2]),
                                          (NpDimension.FRAME_ITER, f[NpDimension.FRAME_ITER]),
                                          (NpDimension.KEYPOINT_ITER, f[NpDimension.KEYPOINT_ITER]),
                                          (NpDimension.KP_DATA, KpDim.accuracy)]))

        bonemat, bones = Bones.getBonemat(f[NpDimension.KEYPOINT_ITER])
        newprediction = boneLengthOptimization._optimize3DPoints(P, P_, prediction, None, bonemat.transpose(), x[0], x[1], x_acc[0], x_acc[1])

        newprediction = np.stack([newprediction[...,0],newprediction[...,1],newprediction[...,2],prediction_acc],axis=2)

        return Sequence(sequence3d.cameras,newprediction, OrderedDict([(NpDimension.FRAME_ITER, f[NpDimension.FRAME_ITER]),
                                                           (NpDimension.KEYPOINT_ITER, f[NpDimension.KEYPOINT_ITER]),
                                                           (NpDimension.KP_DATA, (KpDim.x, KpDim.y, KpDim.z, KpDim.accuracy))]))


    @staticmethod
    def _optimize3DPoints(P : np.array, P_: np.array, prediction, bestEstimate = None, bonemat=None, x1 = None, x2 = None, acc1 = None, acc2=None) -> np.array:
        if bestEstimate is None:
            bestEstimate = np.copy(prediction)

        #    C = np.null_space(P)
        #    C_ = np.null_spacae(P_)

        #    Pinv = np.linalg.pinv()
        #    P_inv = np.linalg.pinv()

        mask = (np.isnan(bestEstimate).any(axis=2).any(axis=1) == False)
        bestEstimate = bestEstimate[mask]
        prediction = prediction[mask]
        if acc1 is None:
            acc1 = np.ones(prediction.shape[0:1])
        if acc2 is None:
            acc2 = np.ones(prediction.shape[0:1])
        x1 = x1[mask]
        x2 = x2[mask]

        bl, bonemat = boneLengthOptimization.bonelength(prediction, bonemat)
        meanbl = np.nanmean(bl, axis=0)

        target_bone = (np.moveaxis(bestEstimate, 1, 2).reshape(-1, 17) @ bonemat).reshape(-1, 3, bonemat.shape[1])
        target_bone *= (meanbl / np.linalg.norm(target_bone, axis=1))[:, None, :]
        target_bone[np.isnan(target_bone)] = 0
        target_bone = np.moveaxis(target_bone, 1, 2)


        bones_ = np.array([np.where(bonemat.transpose() == 1)[1], np.where(bonemat.transpose() == -1)[1]]).transpose()
        bones = cp.Parameter(bones_.shape)
        # acc1_ = cp.Parameter(17, nonneg=True)
        # acc2_ = cp.Parameter(17, nonneg=True)
        bones.value = bones_

        target_bone = np.append(target_bone, np.ones((target_bone.shape[0], target_bone.shape[1], 1)), axis=2)

        prediction = np.append(prediction, np.ones((prediction.shape[0], prediction.shape[1], 1)), axis=2)
        newprediction = np.copy(prediction)

        tb = cp.Parameter(target_bone.shape[1:])
        Px = cp.Parameter((17, 3))
        P_x = cp.Parameter((17, 3))
        x_ = cp.Variable((17, 3))
        X = cp.Parameter((17, 3))

        term = (cp.sum([boneLengthOptimization.projdist(Px[k], P_x[k], x_[k], P, P_) for k in range(17)]) +
                # cp.sum([globalDist(X[k], x_[k]) for k in range(17)]) +
                20 * cp.sum([boneLengthOptimization.bonediff(bones[k], x_, tb[k]) for k in range(bones.shape[0])]))

        obj = cp.Minimize(term)
        prob = cp.Problem(obj)

        frames = prediction.shape[0]

        # for i in range(0,frames):

        def process(i):
            bestEstimatei = bestEstimate[i]

            p1 = (P[:, :3] @ bestEstimatei.transpose() + P[:, 3, None]).transpose()
            if x1 is None:
                p1[:, :2] /= p1[:, 2, None]
            else:
                p1[:, :2] = x1[i]
            p1[:, 2] = 1 / p1[:, 2]  # invert 3rd row prevend division
            # p1 *= acc1[i,:,None] / (acc1[i,:,None] + acc2[i,:,None]) # already weight point
            # here p1 is weighted by the prediction accuracy. Since we also weight p1[2] which is later
            # multiplied to the point we dont need to multiply acc1 later explicitly to the predicted point

            Px.value = p1

            p_1 = (P_[:, :3] @ bestEstimatei.transpose() + P_[:, 3, None]).transpose()
            if x2 is None:
                p_1[:, :2] /= p_1[:, 2, None]
            else:
                p_1[:, :2] = x2[i]
            p_1[:, 2] = 1 / p_1[:, 2]  # invert 3rd row prevend division
            # p_1 *= acc2[i,:,None]  / (acc1[i,:,None] + acc2[i,:,None])  # already weight point
            # here p_1 is weighted by the prediction accuracy. Since we also weight p_1[2] which is later
            # multiplied to the point we dont need to multiply acc2 later explicitly to the predicted point
            P_x.value = p_1

            tb.value = target_bone[i]

            x_.value = bestEstimatei[:, :3]
            X.value = bestEstimatei[:, :3]
            prob.solve()
            if (i % 10 == 0): print(i, "/", prediction.shape[0], end="\r")
            return x_.value
            #

        def wrapper(i):
            try:
                #newprediction[i, :, :3] = process(i)
                return process(i)
            except ValueError:
                print("ValueError. Continue")
                print(traceback.format_exc())
                #newprediction[i, :, :3] = np.nan
                return np.nan

        #for i in range(frames):
        #    newprediction[i, :, :3] = wrapper(i)

        with Pool(processes=24) as pool:
            newprediction[:,:,:3] = np.array(pool.map(wrapper, range(frames)))

        # with Pool(processes=10) as pool:
        #    newprediction[:,:,:3] = np.array(pool.map(process, range(frames)))
        # obj = cp.Minimize(cp.sum(objarray))
        # prob = cp.Problem(obj)
        # prob.solve(verbose=True)

        retarray = np.empty([len(mask), prediction.shape[1], 3])
        retarray[mask == False] = np.nan
        retarray[mask == True] = newprediction[:, :, :3]
        return retarray

    @staticmethod
    def optimizeP(P, P_, prediction, bestEstimate=None, x1=None, x2=None):
        if bestEstimate is None:
            bestEstimate = prediction

        mask = (np.isnan(bestEstimate).any(axis=2).any(axis=1) == False)
        bestEstimate = bestEstimate[mask]
        prediction = prediction[mask]

        estimationRange = list(range(prediction.shape[0]))
        estimationRange = random.choices(estimationRange, k=100)

        prediction = np.append(prediction, np.ones(prediction.shape - np.array([0, 0, 2])), axis=2)

        Px = [cp.Parameter((17, 3)), ] * len(estimationRange)
        X = [cp.Parameter((17, 3)), ] * len(estimationRange)

        # Rquart = cp.Parameter(4,value=[0,0,0,0])

        # w = Rquart[0]
        # x = Rquart[1]
        # y = Rquart[2]
        # z = Rquart[3]

        # R = cp.vstack([cp.hstack([1 - 2*y*y - 2*z*z ,     2*x*y - 2*z*w,     2*x*z + 2*y*w]),
        #                cp.hstack([    2*x*y + 2*z*w , 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w]),
        #                cp.hstack([    2*x*z - 2*y*w ,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y])])#

        R = cp.Parameter((3, 3), value=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        T = cp.Variable(3, value=[0, 0, 0])

        def projdist(p1, x_):
            # p2 = P[:,:3]@R@x_ + P[:,3] + T
            p2 = P[:, :3] @ x_ + P[:, 3] + T
            p2 *= p1[2]

            return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

        term = (cp.sum([
            cp.sum(
                [projdist(Px[i][k], X[i][k]) for k in range(17)]
            ) for i in range(len(estimationRange))
        ]))

        obj = cp.Minimize(term)
        prob = cp.Problem(obj)

        def process(P__, x1_):

            ### First run: P
            for i in range(len(estimationRange)):
                p1 = (P__[:, :3] @ bestEstimate[estimationRange[i]].transpose() + P__[:, 3, None]).transpose()
                if x1_ is None or np.isnan(x1_[estimationRange[i]]).any():
                    p1[:, :2] /= p1[:, 2, None]
                else:
                    p1[:, :2] = x1_[estimationRange[i]]
                p1[:, 2] = 1 / p1[:, 2]  # invert 3rd row prevend division
                Px[i].value = p1

                X[i].value = bestEstimate[estimationRange[i], :, :3]

            prob.solve()

            P__[:, :3] = P__[:, :3] @ R.value
            P__[:, 3] = P__[:, 3] + T.value

            return P__

        newP = process(P, x1)
        newP_ = process(P_, x2)

        return newP, newP_