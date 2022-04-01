from collections import OrderedDict

import cvxpy.error
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import cvxpy as cp

from src.DataInterface.Sequence import Sequence, NpDimension
from src.Skeleton.Bones import Bones
from src.Skeleton.Keypoint import Dimension as KpDim

VISUALIZE = False
KEEP_ALL_RANSAC_INLIER_DATA = False

class EllipseCorrector:
    @staticmethod
    def correct_distortion(sequence: Sequence, goodsamples = 1000, maxstepsfactor = 40, alpha = 1) -> Sequence:
        newprediction, f = sequence.get(format=OrderedDict([(NpDimension.FRAME_ITER, None),
                                                            (NpDimension.KEYPOINT_ITER, None),
                                                            (NpDimension.KP_DATA, (KpDim.x, KpDim.y, KpDim.z))]))

        accuracies, f_ = sequence.get(format=OrderedDict([(NpDimension.FRAME_ITER, None),
                                                            (NpDimension.KEYPOINT_ITER, None),
                                                            (NpDimension.KP_DATA,
                                                             KpDim.accuracy)]))

        bonemat, bones = Bones.getBonemat(f[NpDimension.KEYPOINT_ITER])
        bonemat = bonemat.transpose()
        medianpt = np.nanmedian(newprediction.reshape(-1, 3), axis=0)
        bonevect = np.array([newprediction[..., 0] @ bonemat, newprediction[..., 1] @ bonemat, newprediction[..., 2] @ bonemat])
        meanbonelen = np.nanmedian(np.linalg.norm(bonevect, axis=0), axis=0)

        bonevectaccuracies = np.exp(np.log(accuracies) @ abs(bonemat))

        print("OLDMEAN: {}".format(np.nanmean(np.linalg.norm(bonevect, axis=0), axis=0)))
        print("OLDVAR: {}".format(np.nanvar(np.linalg.norm(bonevect, axis=0), axis=0)))

        # bonevect = bonevect/meanbonelen[None,None,:]
        # boneid = np.array(range(len(meanbonelen))).astype(float)
        # newbonevect = np.empty((4, bonevect.shape[1], bonevect.shape[2]))
        # newbonevect[:3] = bonevect
        # newbonevect[3] = boneid[None, :]
        newbonevect = bonevect
        # bonevect = newbonevect.reshape(4, -1)
        # bonevect = bonevect[:, ~np.isnan(bonevect).any(axis=0)]

        data = np.empty((newbonevect.shape[2], newbonevect.shape[1], 5 + len(meanbonelen)))

        data[:, :, :5] = np.array(
            [np.square(newbonevect[0]) + np.square(newbonevect[1]) - 2 * np.square(newbonevect[2]),
             np.square(newbonevect[0]) - 2 * np.square(newbonevect[1]) + np.square(newbonevect[2]),
             4 * newbonevect[0] * newbonevect[1],
             2 * newbonevect[0] * newbonevect[2],
             2 * newbonevect[1] * newbonevect[2],
             ]).transpose()
        data[:, :, 5:] = 0
        for i in range(bonemat.shape[1]):
            data[i, :, 5 + i] = 1
        e = np.array([np.square(bonevect[0]) + np.square(bonevect[1]) + np.square(bonevect[2])]).transpose()

        samplesize = 10  # plus one of each bone for scale
        ransac_steps = goodsamples
        sample_ranges = [list(np.array(range(k.shape[0]))[~np.isnan(k).any(axis=1)]) for k in bonevect.swapaxes(0, 2)]
        # inliers_threshold = 0.0005
        inliers_threshold = 0.05

        samples = list()
        for _ in range(ransac_steps * maxstepsfactor):
            choices = np.array(random.choices(range(data.shape[0]), k=samplesize))
            samples.append([random.sample(sample_ranges[i], (choices == i).sum() + 1) for i in range(bonevect.shape[2])])
        data_param = cp.Parameter((data.shape[0] + samplesize, data.shape[2]))
        e_param = cp.Parameter((data.shape[0] + samplesize, 1))

        s = cp.Variable((5 + len(meanbonelen), 1))

        obj_fun = cp.sum(cp.abs(data_param @ s - e_param))

        # constraint = [cp.sum(s[5:]) == bonemat.shape[1],]
        constraint = []
        objective = cp.Minimize(obj_fun)
        prob = cp.Problem(objective=objective, constraints=constraint)

        if (KEEP_ALL_RANSAC_INLIER_DATA):
            ransac_inliers = list()
        else:
            maxdata = None
            count_steps = 0

        count_over_treshold = 0
        maxsum = 0

        for i in range(ransac_steps * maxstepsfactor):

            sample = samples[i]

            data_param.value = np.array(np.vstack([data[a,sample[a]] for a in range(len(sample))]))
            e_param.value = np.array(np.vstack([e[a,sample[a]] for a in range(len(sample))]))
            #s.value = np.array(np.hstack([np.ones(len(sample[a])) * meanbonelen[a] for a in range(len(sample))]))
            try:
                result = prob.solve(warm_start = True,verbose=False)
            except cvxpy.error.SolverError as err:
                print(err);
                continue

            abs_error = np.abs(data.reshape(-1, data.shape[2]) @ s.value - e.reshape(-1, 1)).squeeze()
            abs_error = abs_error.reshape(data.shape[0], data.shape[1])
            #abs_error /= np.sqrt(abs(s.value[5:]))
            abs_error /= abs(s.value[5:])
            abs_error = abs_error.reshape(-1)

            # fittness = 1-(AbsError/inliers_threshold)
            # fittness[fittness<0] = 0

            new_ransac_inliers = (abs_error < inliers_threshold) * bonevectaccuracies.transpose().reshape(-1)

            if (KEEP_ALL_RANSAC_INLIER_DATA):  # keep all ransac
                ransac_inliers.append(new_ransac_inliers)
            count_steps += 1



            goodsample_tresh = new_ransac_inliers.shape[0] * 0.25
            # ransacInliers.append(fittness)
            thissum = np.nansum(new_ransac_inliers)

            if thissum > maxsum:
                maxsum = thissum
                maxdata = new_ransac_inliers
            if thissum > (goodsample_tresh):
                count_over_treshold += 1
                if count_over_treshold >= ransac_steps:
                    break
            if i%1000 == 0:
                print("{} samples analyzed, {} good samples found. Best has inlier-score {}".format(count_steps, count_over_treshold,maxsum/goodsample_tresh))
        print("{} samples analyzed, {} good samples found.".format(count_steps, count_over_treshold))

        # print(np.nanmax(AbsError), np.nanmedian(AbsError),ransacInliers[-1])


        #amax = np.argmax(np.sum(np.array(ransac_inliers), axis=1))
        #data_ = data.reshape(-1, data.shape[2])[ransac_inliers[amax] > 1e-6]
        #e_ = e.reshape(-1, 1)[ransac_inliers[amax] > 1e-6]
        data_ = data.reshape(-1, data.shape[2])[maxdata > 1e-6]
        e_ = e.reshape(-1, 1)[maxdata > 1e-6]

        result = cp.Problem(cp.Minimize(cp.sum(cp.abs(data_ @ s - e_))), constraints=constraint).solve()

        # sample ellipse

        s = s.value.squeeze()
        Aa = (1 - s[0] - s[1]) / 3 #(1-U-V/3)
        Ab = Aa + s[1]             #(Aa + V)
        Ac = Aa + s[0]             #(Aa + U)
        Ad = -4 * s[2] / 6         #(-4 * M / 6)
        Ae = -2 * s[3] / 6         #(-2 * N / 6)
        Af = -2 * s[4] / 6         #(-2 * P / 6)

        A = np.array([[Aa, Ad, Ae],
                      [Ad, Ab, Af],
                      [Ae, Af, Ac]])

        A_ = np.linalg.eig(A)

        npr = newprediction.reshape(-1, 3)
        # npr = A_[1] @ npr.transpose()
        npr = np.linalg.inv(A_[1]) @ npr.transpose()
        sv = np.sqrt(np.abs(A_[0][:, None]))
        sv /= np.mean(sv)
        npr *= (sv * alpha) + (np.array([[1,],[1,],[1,]]) * (1-alpha))
        npr = A_[1] @ npr
        # npr = np.linalg.inv(A_[1]) @ npr
        npr_ = npr.transpose().reshape(newprediction.shape)
        bonevect_ = np.array([npr_[..., 0] @ bonemat, npr_[..., 1] @ bonemat,
                              npr_[..., 2] @ bonemat])
        print("NEWMEAN: {}".format(np.nanmean(np.linalg.norm(bonevect_, axis=0), axis=0)))
        print("NEWVAR: {}".format(np.nanvar(np.linalg.norm(bonevect_, axis=0), axis=0)))
        print("A =\n"+str(A/np.sum(A)*3))
        new_medianpt = np.nanmedian(npr_.reshape(-1, 3), axis=0)

        npr_ += (medianpt - new_medianpt)[None, None, :]
        #npr_ -= new_medianpt[None,None,:]



        if(VISUALIZE):
            randomData = np.random.random((2, 2000)) * 16 - 8
            a__ = A[2, 2]
            b__ = randomData[0] * (A[0, 2] + A[2, 0]) + randomData[1] * (A[1, 2] + A[2, 1])
            c__ = (randomData[0] ** 2) * (A[0, 0]) + randomData[0] * randomData[1] * (A[0, 1] + A[1, 0]) + (
                    randomData[1] ** 2) * A[1, 1] - 1

            x__ = (-b__ + np.sqrt(b__ ** 2 - 4 * a__ * c__)) / (2 * a__)
            x2__ = (-b__ - np.sqrt(b__ ** 2 - 4 * a__ * c__)) / (2 * a__)
            ellipse = np.empty((3, 2 * (randomData.shape[1])))
            ellipse[:2, :randomData.shape[1]] = randomData
            ellipse[:2, randomData.shape[1]:] = randomData
            ellipse[2, :randomData.shape[1]] = x__
            ellipse[2, randomData.shape[1]:] = x2__
            matplotlib.use("TKAgg")
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim3d([-0.2, 0.2])
            ax.set_ylim3d([-0.2, 0.2])
            ax.set_zlim3d([-0.2, 0.2])
            x = (bonevect[0, :, :] / np.sqrt(s[None, 5:])).flatten()
            y = (bonevect[1, :, :] / np.sqrt(s[None, 5:])).flatten()
            z = (bonevect[2, :, :] / np.sqrt(s[None, 5:])).flatten()
            # x_ = bonevect_[0,:,2]
            # y_ = bonevect_[1,:,2]
            # z_ = bonevect_[2,:,2]
            x__ = ellipse[0]
            y__ = ellipse[1]
            z__ = ellipse[2]
            data, = ax.plot(x, y, z, linestyle="", marker="o", color="blue")
            # data2, = ax.plot(x_,y_,z_, linestyle="", marker = "o", color="red")
            data3, = ax.plot(x__, y__, z__, linestyle="", marker="o", color="yellow")
            plt.show(block=False)
            fig.canvas.draw()

        npr_ =  np.stack([npr_[..., 0], npr_[..., 1], npr_[..., 2], accuracies], axis=2)
        return Sequence(sequence.cameras,npr_,OrderedDict([(NpDimension.FRAME_ITER, f[NpDimension.FRAME_ITER]),
                                                           (NpDimension.KEYPOINT_ITER, f[NpDimension.KEYPOINT_ITER]),
                                                           (NpDimension.KP_DATA, (KpDim.x, KpDim.y, KpDim.z, KpDim.accuracy))])), A