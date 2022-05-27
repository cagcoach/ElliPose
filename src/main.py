from pickle import HIGHEST_PROTOCOL
import compress_pickle as pickle

from collections import OrderedDict
from copy import deepcopy, copy
from shutil import copyfile

import math
import sys, os
#from multiprocessing.pool  import ThreadPool as Pool
from pathos.multiprocessing import ProcessingPool as Pool
import random
import numpy as np
import scipy
import scipy.linalg
import scipy.ndimage
import matplotlib.pyplot

from src.DataInterface.CPN2D import CPN2D
from src.DataInterface.Camera import Camera
from src.DataInterface.Human36mGT import Human36mGT
from src.DataInterface.Human36mGT2D import Human36mGT2D
from src.DataInterface.NoTears import NoTears
from src.DataInterface.Sequence import NpDimension, Sequence
from src.Estimated3dDataset import Estimated3dDataset
from src.Skeleton.CommonKeypointFormats import COMMON_KEYPOINT_FORMATS
from src.boneLengthOptimization import boneLengthOptimization
from src.Skeleton.Bones import Bones
from src.h36m_noTears import Human36mNoTears
from src.EllipseCorrector import EllipseCorrector
import traceback
from src.Skeleton.Keypoint import Dimension as KpDim
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath('external/VideoPose3D'))
from external.VideoPose3D.common.camera import normalize_screen_coordinates, camera_to_world, world_to_camera
from external.VideoPose3D.common.loss import p_mpjpe
from external.VideoPose3D.common.h36m_dataset import Human36mDataset
import cv2

from configparser import ConfigParser

def __OLD__points2Dto3D(points1, points2):
    #From Hartley etal. - Miltiple View Geometry in Computer Vision, Page 318

    fundmat = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, ransacReprojThreshold=0.001, confidence=0.999, maxIters=5*points1.shape[0])

    #points1 = points1[fundmat[1].squeeze()==1]
    #points2 = points2[fundmat[1].squeeze() == 1]

    for i in range(points1.shape[0]):

        p1 = points1[i]
        p2 = points2[i]

        # (i)  Define transformation matrices
        T = np.array([[1,0,-p1[0]],
                      [0,1,-p1[1]],
                      [0,0,1]]).astype(np.double)
        T_ = np.array([[1, 0, -p2[0]],
                      [0, 1, -p2[1]],
                      [0, 0, 1]]).astype(np.double)

        # (ii) Replace F by T'^{−T}FT^{−1}. The new F corresponds to translated coordinates.
        F = np.linalg.inv(T_).transpose() @ np.copy(fundmat[0]) @ np.linalg.inv(T)

        # (iii) Compute the right and left epipoles
        e = scipy.linalg.null_space(F)
        e_ = scipy.linalg.null_space(F.transpose())
        #print(e,e_)
        try:
            e  /= math.sqrt(e[0,0] ** 2 + e[1,0] ** 2)
            e_ /= math.sqrt(e_[0,0] ** 2 + e_[1,0] ** 2)
        except IndexError:
            continue

        # (iv) Form matrices R and R_
        R = np.array([[e[0,0],e[1,0],0],
                      [-e[1,0],e[0,0],0],
                      [0,0,1]])
        R_ = np.array([[e_[0,0], e_[1,0], 0],
                      [-e_[1,0], e_[0,0], 0],
                      [0, 0, 1]])

        # (v) Replace F by R' F R^T
        F = R_ @ F @ R.transpose()

        # (vi) Set f = e3, f' = e'3, a = F22, b = F23, c = F32 and d = F33
        f = e[2,0]
        f_ = e_[2,0]
        a = F[1,1]
        b = F[1,2]
        c = F[2,1]
        d = F[2,2]

        f2 = f ** 2
        f_2 = f_ ** 2
        a2 = a ** 2
        b2 = b ** 2
        c2 = c ** 2
        d2 = d ** 2

        a3 = a ** 3
        b3 = b ** 3
        c3 = c ** 3
        d3 = d ** 3

        f4 = f ** 4
        f_4 = f_ ** 4
        a4 = a ** 4
        b4 = b ** 4
        c4 = c ** 4
        d4 = d ** 4

        #print(F)
        roots = np.array([
                        ( - a2 * c * d * f4 + a * b * c2 * f4),
                        (a4 + 2 * a2 * c2 * f_2 - a2 * d2 * f4 + b2 * c2 * f4 + c4 * f_4),
                        (4 * a3 * b - 2 * a2 * c * d * f2 + 4 * a2 * c * d * f_2 + 2 * a * b * c2 * f2 + 4 * a * b * c2 * f_2 - a * b * d2 * f4 + b2 * c * d * f4 + 4 * c3 * d * f_4),
                        (6 * a2 * b2 - 2 * a2 * d2 * f2 + 2 * a2 * d2 * f_2 + 8 * a * b * c * d * f_2 + 2 * b2 * c2 * f2 + 2 * b2 * c2 * f_2 + 6 * c2 * d2 * f_4),
                        ( - a2 * c * d + 4 * a * b3 + a * b * c2 - 2 * a * b * d2 * f2 + 4 * a * b * d2 * f_2 + 2 * b2 * c * d * f2 + 4 * b2 * c * d * f_2 + 4 * c * d3 * f_4),
                        ( - a2 * d2 + b4 + b2 * c2 + 2 * b2 * d2 * f_2 + d4 * f_4),
                        ( - a * b * d2 + b2 * c * d)
        ])
        t0 = np.roots(roots)


        #(viii) Evaluate the cost function

        def s(t):
            return (t ** 2) / (1 + f2 * t ** 2) + (c * t + d) ** 2 / ((a * t + b) ** 2 + f_2 * (c * t + d) ** 2)

        #test infinity
        tmin = 1e50
        stmin = s(1e50)

        #test neg-infinity
        st = s(-1e50)
        if st < stmin:
            tmin = -1e50
            stmin = st

        #test all t0
        for t in t0:
            st = s(t.real)
            if st < stmin:
                tmin = t.real
                stmin = st

        ltmin = np.array([tmin*f,1,-tmin])
        l_tmin = F @ np.array([[0],[tmin],[1]]).squeeze()

        def closestToOrigin(l):
            return np.array([-l[0]*l[2],-l[1]*l[2], l[0]**2+l[1]**2])

        xhat = closestToOrigin(ltmin)
        xhat_ = closestToOrigin(l_tmin)

        xhat = np.linalg.inv(T) @ R.transpose() @ xhat
        xhat_ = np.linalg.inv(T_) @ R_.transpose() @ xhat_

        xhat /= xhat[2]
        xhat_ /= xhat_[2]

        points1[i] = xhat[:2]
        points2[i] = xhat_[:2]

    P = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0]])

    #e = scipy.linalg.null_space(fundmat[0])
    e_ = scipy.linalg.null_space(fundmat[0].transpose())

    P_ = np.append(np.cross(e_.squeeze(), fundmat[0]),e_,axis=1)
    res = cv2.triangulatePoints(P,P_,points1.transpose(),points2.transpose()).transpose()
    return res / res[:,3]

def estimateForKeypoint(var,iterellipse,iterbone,config,filenameappend) -> (Sequence, Sequence, np.array, np.array,ConfigParser):

    s = var[0]
    k = var[1]

    cam1_idx = var[3]
    cam2_idx = var[4]
    #gt = var[5]
    outpath = var[6]

    picklepath = os.path.join(outpath, "{}_{}{}.pkl".format(s, k, filenameappend))
    picklepaths = [picklepath + a for a in ["", ".gz",".bz",".lzma",".zip"]]
    picklepath = picklepath + ".lzma"
    try:
        for pp in picklepaths:
            if(os.path.exists(pp)):
                print(s + " " + k + " already exists: loading from file " + pp)
                with open(pp, "rb") as input_file:
                    datadict = pickle.load(input_file)
                #print("loaded")
                #{"sequence3D":sequence3d, "P":P, "P_":P_, "pred":pred,"newpred":newpred, "action":k, "subject":s, "bonemat": bonemat, "bones": bones}

                if not "aligned3d" in datadict:
                    datadict["aligned3d"] = Sequence(datadict["sequence3D"].cameras,
                                                     datadict["newpred"],
                                                     OrderedDict([(NpDimension.FRAME_ITER,datadict["sequence3D"].npdimensions[NpDimension.FRAME_ITER]),
                                                                  (NpDimension.KEYPOINT_ITER,datadict["sequence3D"].npdimensions[NpDimension.KEYPOINT_ITER]),
                                                                  (NpDimension.KP_DATA,(KpDim.x, KpDim.y, KpDim.z)),
                                                                  ]))
                    with open(pp, 'wb') as f:
                        pickle.dump(datadict, f)

                return datadict["sequence3D"], datadict["aligned3d"], datadict["P"], datadict["P_"]
                #return sequence3d, P, P_
                #return pred, newpred, gt, k, s, P, P_
    except:
        pass
    try:
        sequence2d = var[2].get_sequence(s, k)
    except IndexError as e:
        print(str(e))
        return None, None, None

    prediction, P, P_ = sequence2d.triangulate3DPoints(cam1_idx,cam2_idx, blur = False, useExtrinsics=False)
    #prediction, _, _ = sequence2d.triangulate3DPoints(cam1_idx,cam2_idx, blur = False, useExtrinsics=False)
    GTCameraDist = sequence2d.cameras[sequence2d.cameras.index(cam1_idx)].translation - sequence2d.cameras[
        sequence2d.cameras.index(cam2_idx)].translation
    GTCameraDist = np.linalg.norm(GTCameraDist)
    gtsequence = var[5].get_sequence(s,k)
    c1gt = sequence2d.cameras[sequence2d.cameras.index(cam1_idx)]
    c2gt = sequence2d.cameras[sequence2d.cameras.index(cam2_idx)]

    sequence3d = copy(prediction)
    c1 = sequence3d.cameras[sequence3d.cameras.index(cam1_idx)]
    c2 = sequence3d.cameras[sequence3d.cameras.index(cam2_idx)]
    #P = None
    #P_ = None

    #prediction = camera_to_world((prediction.reshape(-1, 3) * realCameraDist).astype(float),
    #                             c1.orientation.astype(float), c1.translation).reshape(-1, 17, 3)

    #print("P-MPJPE: {} {:.2f} mm ({} frames)".format(k,
    #                                                 p_mpjpe((prediction[:, :, :]) * np.array([1, 1, 1]),
    #                                                         (gt[:,:, :])) * 1000, gt.shape[0]))

    for i in range(0):
        #sequence3d = optimize2DPointLocationUsingMeanBoneLength(P, P_, prediction, bestEstimate=sequence3d,
        #                                                           bonemat=bonemat, x1=kp1, x2=kp2, acc1=acc1,
        #                                                           acc2=acc2)
        pass
    bestA = np.inf
    bestSequence3D = None
    bestP = None
    bestP_ = None
    iterationCounter = 0
    while True:#iterellipse>0: # iterellipse is a const! Thus this is either always true or always false. See also breaking conditions below
        corrected_sequence3d, A, inliers = EllipseCorrector.correct_distortion(
            sequence3d,
            goodsamples=config.getint("Ellipse","goodsamples"),
            maxstepsfactor=config.getint("Ellipse","maxfactor"),
            alpha=config.getfloat("Ellipse","alpha"),
            samplesize=config.getint("Ellipse","samplesize"),
            symmetricLengh=config.getboolean("Ellipse","symmetricLength"),
            inliers_threshold=config.getfloat("Ellipse","inlierThreshold"),
        )

        evaluatedA = np.sum(np.square((A / np.sum(A) * 3) - np.eye(3)))
        print("evaluatedA = " + str(evaluatedA))

        if evaluatedA < config.getfloat("Ellipse", "breakingcondition"):
            bestA = evaluatedA
            print("IT'S A BALL! breaking condition met")
            #sequence3d = corrected_sequence3d
            break
        if bestA > evaluatedA:
            bestA = evaluatedA
            #bestSequence3D = corrected_sequence3d
            bestSequence3D = sequence3d
            bestP = P
            bestP_ = P_
        if iterationCounter >= iterellipse:
            if config["Ellipse"].getboolean("SelectBestA"):
                sequence3d = bestSequence3D
                P = bestP
                P_ = bestP_
            else:
                # sequence3d = corrected_sequence3d
                bestA = evaluatedA
            break
        iterationCounter += 1

        sequence3d = corrected_sequence3d

        nprNp, nprFormat = sequence3d.get(format=[(NpDimension.FRAME_ITER, None),
                         (NpDimension.KEYPOINT_ITER, None),
                         (NpDimension.KP_DATA, (KpDim.x, KpDim.y, KpDim.z))])

        accNp, accFormat = sequence3d.get(format=[(NpDimension.FRAME_ITER, nprFormat[NpDimension.FRAME_ITER]),
                                           (NpDimension.KEYPOINT_ITER, nprFormat[NpDimension.KEYPOINT_ITER]),
                                           (NpDimension.KP_DATA, KpDim.accuracy)])

        kp, f = sequence2d.get(format=[(NpDimension.CAM_ITER, [cam1_idx,cam2_idx]),
                                  (NpDimension.FRAME_ITER, nprFormat[NpDimension.FRAME_ITER]),
                                  (NpDimension.KEYPOINT_ITER, nprFormat[NpDimension.KEYPOINT_ITER]),
                                  (NpDimension.KP_DATA, (KpDim.x, KpDim.y))])
        kpacc, _ = sequence2d.get(format=[(NpDimension.CAM_ITER, [cam1_idx, cam2_idx]),
                                     (NpDimension.FRAME_ITER, nprFormat[NpDimension.FRAME_ITER]),
                                     (NpDimension.KEYPOINT_ITER, nprFormat[NpDimension.KEYPOINT_ITER]),
                                     (NpDimension.KP_DATA, KpDim.accuracy)])


        c1 = sequence3d.cameras[sequence3d.cameras.index(cam1_idx)]
        c2 = sequence3d.cameras[sequence3d.cameras.index(cam2_idx)]

        #nanmask = np.isnan(nprNp.reshape(-1,3)[accNp.reshape(-1)>0.7]).any(axis=1)

        #mask = accNp.reshape(-1)>0.7
        #mask &= ~np.isnan(nprNp.reshape(-1,3)).any(axis=1)

        bm = Bones.getBonemat(f[NpDimension.KEYPOINT_ITER])
        mask = (inliers @ np.abs(bm[0])) > 1e-6
        R1 = cv2.solvePnPRansac(objectPoints = nprNp[mask],
                                imagePoints = kp[0][mask],
                                cameraMatrix = c1.intrinsicMatrix,
                                distCoeffs=(0, 0, 0, 0),
                                rvec = (cv2.Rodrigues(src=c1.rotationMatrix.copy())[0]),
                                tvec = c1.translation.reshape((1,3)).copy(),
                                useExtrinsicGuess=True,
                                confidence = config.getfloat("Ellipse","PnPconfidence"),#0.99
                                reprojectionError=config.getfloat("Ellipse","PnPreprojectionError"),#0.05,
                                iterationsCount = config.getint("Ellipse","PnPiterations")) # 10000

        R2 = cv2.solvePnPRansac(objectPoints=nprNp[mask],
                                imagePoints=kp[1][mask],
                                cameraMatrix=c2.intrinsicMatrix,
                                distCoeffs=(0, 0, 0, 0),
                                rvec=(cv2.Rodrigues(src=c2.rotationMatrix.copy())[0]),
                                tvec=c2.translation.reshape((1,3)).copy(),
                                useExtrinsicGuess=True,
                                confidence = config.getfloat("Ellipse","PnPconfidence"),#0.99
                                reprojectionError=config.getfloat("Ellipse","PnPreprojectionError"),
                                iterationsCount = config.getint("Ellipse","PnPiterations")) # 10000

        #shape = nprNp.shape

        #nprNp = nprNp.reshape(-1, 3).transpose()
        #npr_ = np.empty((4,nprNp.shape[1]))
        #npr_[:3,:] = nprNp.copy()
        #npr_[3,:] = 1

        Rt1 = np.append(np.append(cv2.Rodrigues(R1[1])[0], R1[2], axis=1), [[0,0,0,1]], axis=0)
        Rt2 = np.append(np.append(cv2.Rodrigues(R2[1])[0], R2[2], axis=1), [[0,0,0,1]], axis=0)


        #npr_ = Rt1 @ npr_

        Rt2_ = Rt2 @ np.linalg.inv(Rt1)
        Rt1_ = np.eye(4)

        scale = 1/np.linalg.norm(Rt2_[:3,3])
        if scale == np.nan:
            raise ValueError()

        #npr_[:3] = npr_[:3] * scale
        Rt2_[:3,3] *= scale

        c1.translation = Rt1_[:3, 3]
        c2.translation = Rt2_[:3, 3]
        c1.rotationMatrix = Rt1_[:3, :3]
        c2.rotationMatrix = Rt2_[:3, :3]

        P =  c1.intrinsicMatrix @ Rt1_[:3]
        P_ = c2.intrinsicMatrix @ Rt2_[:3]
        #npr_ = npr_[:3].transpose().reshape(shape)

        '''
        if evaluatedA < config.getfloat("Ellipse", "breakingcondition"):
            print("IT'S A BALL! breaking condition met")
            #sequence3d = corrected_sequence3d
            sequence3d._cameras = [c1, c2]
            break
        if bestA > evaluatedA:
            bestA = evaluatedA
            sequence3d._cameras = [c1,c2]
            bestSequence3D = sequence3d
            bestP = P
            bestP_ = P_
        if iterationCounter >= iterellipse:
            if config["Ellipse"].getboolean("SelectBestA"):
                sequence3d = bestSequence3D
                P = bestP
                P_ = bestP_
            break
        iterationCounter += 1
        '''
        triangulatedPoints = cv2.triangulatePoints(P, P_, kp[0].reshape(-1,2).transpose(), kp[1].reshape(-1,2).transpose())
        triangulatedPoints /= triangulatedPoints[3, :]
        triangulatedPoints = triangulatedPoints.transpose()[:, :3].reshape(-1, 17, 3)

        #sequence3d = np.copy(triangulatedPoints)
        nparraynpr = np.stack([triangulatedPoints[...,0],triangulatedPoints[...,1],triangulatedPoints[...,2],accNp],axis=2)
        sequence3d = Sequence([c1,c2],nparraynpr, npdimensions = OrderedDict([
            (NpDimension.FRAME_ITER, nprFormat[NpDimension.FRAME_ITER]),
            (NpDimension.KEYPOINT_ITER, nprFormat[NpDimension.KEYPOINT_ITER]),
            (NpDimension.KP_DATA, (KpDim.x, KpDim.y, KpDim.z, KpDim.accuracy))
        ]))

        #newprediction_WORLD = camera_to_world((sequence3d.reshape(-1, 3) * realCameraDist).astype(float),
        #                                      c1.orientation.astype(float), c1.translation).reshape(-1, 17, 3)
        #sequence3d = npr_
        #print("P-MPJPE: {} {:.2f} mm ({} frames) iter 0".format(k,
        #                                                         p_mpjpe(
        #                                                             (newprediction_WORLD[:, :, :]) * np.array(
        #                                                                 [1, 1, 1]),
        #                                                             (gt[:, :, :])) * 1000,
        #                                                         gt.shape[0]))



    for i in range(iterbone): #5
        sequence3d = boneLengthOptimization.optimize3DPoints(P,P_,sequence2d,sequence3d,cam1_idx,cam2_idx)

        #sequence3d = boneLengthOptimization.optimize3DPoints(P, P_, triangulatedPoints, bestEstimate=sequence3d,
        #                                                           bonemat=bonemat, x1=kp1, x2=kp2, acc1=acc1,
        #                                                           acc2=acc2)
        #newprediction_WORLD = camera_to_world((sequence3d.reshape(-1, 3) * realCameraDist).astype(float),
        #                                c1.orientation.astype(float), c1.translation).reshape(-1, 17, 3)
        #print("P-MPJPE: {} {:.2f} mm ({} frames) iter {}".format(k,
        #                                                         p_mpjpe(
        #                                                             (newprediction_WORLD[:, :, :]) * np.array([1, 1, 1]),
        #                                                              (gt[:, :, :])) * 1000,
        #                                                         gt.shape[0],
        #                                                         i+1))

    newpred, f = sequence3d.get(format=OrderedDict([(NpDimension.FRAME_ITER, None),
                                              (NpDimension.KEYPOINT_ITER, None),
                                              (NpDimension.KP_DATA, (KpDim.x, KpDim.y, KpDim.z))]))
    pred, f_ = prediction.get(format=OrderedDict([(NpDimension.FRAME_ITER, None),
                                                    (NpDimension.KEYPOINT_ITER, None),
                                                    (NpDimension.KP_DATA, (KpDim.x, KpDim.y, KpDim.z))]))

    #newpred_acc, f_acc = sequence3d.get(format=[(NpDimension.FRAME_ITER, nprFormat[NpDimension.FRAME_ITER]),
    #                                          (NpDimension.KEYPOINT_ITER, nprFormat[NpDimension.KEYPOINT_ITER]),
    #                                          (NpDimension.KP_DATA, KpDim.accuracy)])
    c1 = sequence3d.cameras[sequence3d.cameras.index(cam1_idx)]
    bonemat, bones = Bones.getBonemat(f[NpDimension.KEYPOINT_ITER])
    newpred = world_to_camera((newpred.reshape(-1,3) * GTCameraDist).astype(float),c1.orientation.astype(float),c1.translation)
    newpred = camera_to_world(newpred,c1gt.orientation.astype(float),c1gt.translation).reshape(-1,17,3)

    #al_prediction = align(prediction,gt)
    #n_prediction = align(sequence3d,gt)

    #print("### GT ###")
    #gt_length, _ = boneLengthOptimization.bonelength(gt, bonemat)
    #print("### Prediction ###")
    pred_length, _ = boneLengthOptimization.bonelength(pred, bonemat.transpose())

    new_length, _ = boneLengthOptimization.bonelength(newpred, bonemat.transpose())

    np.set_printoptions(precision=3, linewidth = 300)
    '''
    print("MEAN: ",s," ",k)
    #print(np.nanmean(gt_length, axis=0))
    print(np.nanmean(pred_length, axis=0))
    print(np.nanmean(new_length, axis=0))
    print("MEDIAN: ",s," ",k)
    #print(np.nanmedian(gt_length, axis=0))
    print(np.nanmedian(pred_length, axis=0))
    print(np.nanmedian(new_length, axis=0))
    '''
    print("VARIANCE: ",s," ",k)
    #print(np.nanvar(gt_length, axis=0))
    print(np.nanvar(pred_length, axis=0))
    print(np.nanvar(new_length, axis=0))
    #if "Sitting 2" in sequence3d.name:
    #sequence3d.visualizeSingleFrame({NpDimension.FRAME_ITER: 50,
    #                                 NpDimension.KEYPOINT_ITER: None,
    #                                 NpDimension.KP_DATA: (KpDim.x, KpDim.y, KpDim.z)})

    # = np.mean(gt_length, axis=0)



    if False:
        matplotlib.use("TKAgg")
        fig, ax1 = matplotlib.pyplot.subplots()
        #bins = np.linspace(gtmean[9]*0.8, gtmean[9]*1.2, 50)
        #ax1.hist(gt_length[:, 9], bins=bins, alpha=0.5, label='GT')
        #ax1.hist(pred_length[:, 9], bins=bins, alpha=0.5, label='PRED')
        #ax1.hist(new_length[:, 9], bins=bins, alpha=0.5, label='NEW')


        #ax1.set_ylim(top=50.)
        matplotlib.pyplot.legend(loc='upper right')
        matplotlib.pyplot.show()

        matplotlib.use("TKAgg")
        fig = matplotlib.pyplot.figure()
        ax2 = fig.add_subplot(111, projection="3d")
        ax2.scatter(prediction[:, 16, 0] - prediction[:, 15, 0]-(np.linalg.norm(gt[:, 16, 0] - gt[:, 15, 0]) * (prediction[:, 16, 0] - prediction[:, 15, 0]) / np.linalg.norm(prediction[:, 16, 0] - prediction[:, 15, 0])),
                    prediction[:, 16, 1] - prediction[:, 15, 1]-(np.linalg.norm(gt[:, 16, 1] - gt[:, 15, 1]) * (prediction[:, 16, 1] - prediction[:, 15, 1]) / np.linalg.norm(prediction[:, 16, 1] - prediction[:, 15, 1])),
                    prediction[:, 16, 2] - prediction[:, 15, 2]-(np.linalg.norm(gt[:, 16, 2] - gt[:, 15, 2]) * (prediction[:, 16, 2] - prediction[:, 15, 2]) / np.linalg.norm(prediction[:, 16, 2] - prediction[:, 15, 2])))
        matplotlib.pyplot.xlim([-0.5, 0.5])
        matplotlib.pyplot.ylim([-0.5, 0.5])
        ax2.set_zlim([-0.5, 0.5])
        matplotlib.pyplot.show()
    #'''
    #print("P-MPJPE: {} {:.2f} mm ({} frames)".format(k,
    #    p_mpjpe((pred[:, :, :]) * np.array([1, 1, 1]), (gt[:, :, :])) * 1000,gt.shape[0]))

    #print("new P-MPJPE: {} {:.2f} mm ({} frames)".format(k,
    #                                                 p_mpjpe((newpred[:, :, :]) * np.array([1, 1, 1]),
    #                                                         (gt[:, :, :])) * 1000, gt.shape[0]))
    #'''
    gt = None
    #os.remove("{}_{}.running".format(k, s))
    aligned3d = Sequence(sequence3d.cameras,
                                     newpred,
                                     OrderedDict([(NpDimension.FRAME_ITER,
                                                   sequence3d.npdimensions[NpDimension.FRAME_ITER]),
                                                  (NpDimension.KEYPOINT_ITER,
                                                   sequence3d.npdimensions[NpDimension.KEYPOINT_ITER]),
                                                  (NpDimension.KP_DATA, (KpDim.x, KpDim.y, KpDim.z)),
                                                  ]))
    with open(picklepath, 'wb') as f:
        pickle.dump({
                    "sequence3D":sequence3d,
                    "aligned3d": aligned3d,
                    "P":P,
                    "P_":P_,
                    "pred":pred,
                    "newpred":newpred,
                    "action":k,
                    "subject":s,
                    "bonemat": bonemat,
                    "bones": bones,
                    "bestA" : bestA,
                    }, f)

    return sequence3d, aligned3d, P, P_
    # return pred, newpred, gt, k, s, P, P_
    #break; #FIXME Debug out

# ███╗   ███╗ █████╗ ██╗███╗   ██╗
# ████╗ ████║██╔══██╗██║████╗  ██║
# ██╔████╔██║███████║██║██╔██╗ ██║
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝

def main(argv):
    config = ConfigParser()
    cont = False
    if "--continue" in argv[1:]:
        argv.remove("--continue")
        cont = True
    #if len(argv) > 2:
    #    print("Wrong number of arguments")
    #    exit()
    if len(argv) == 1:
        import easygui
        argv.append(easygui.fileopenbox(default=os.path.join(os.path.dirname(sys.argv[0]), "*.conf"),filetypes=[["*.conf", "Configuration File"]], multiple=True))
    ret = config.read(argv[1:])
    if ret != argv[1:]:
        print("Failed reading config files: \n" + "\n".join(list(set(argv[1:]).difference(set(ret)))))
        exit()

    if cont:
        outpath = config["exec"]["outpath"]
        outpath = os.path.abspath(outpath)
        curtimestr = config["exec"]["time"]
    else:
        now = datetime.now()
        curtimestr = now.strftime("%Y%m%d_%H%M%S")
        outpath = "output_{}".format(curtimestr)
        outpath = os.path.abspath(outpath)
        config["exec"]["outpath"] = "output_{}".format(curtimestr)
        config["exec"]["time"] = curtimestr
    if not "RandomSeed" in config["exec"].keys():
        config["exec"]["RandomSeed"] = str(random.randrange(sys.maxsize))
    random.seed(config.getint("exec","RandomSeed"))

    Path(outpath).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(outpath,"config.conf"), 'w') as configfile:
        config.write(configfile)

    #source = "Human3.6m"
    source = config["exec"]["Source"]
    cameraExtrinsicsKnown = config["exec"].getboolean("CameraExtrinsicsKnown")
    cameraIntrinsicsKnown = config["exec"].getboolean("CameraIntrinsicsKnown")
    blur = config["exec"].getboolean("Blur")

    defaultFocalLengthIn35mmEquivalent = 10 #typical smartphone: wide-angle between 22mm and 30mm

    #dataset = Human36mDataset(config.get(source, "3D"))
    human36mGT = Human36mGT(config)

    cam1_idx = config["exec"]["CameraIndex1"]
    cam2_idx = config["exec"]["CameraIndex2"]

    if source == "NoTears":
        poseset = "2D"
        input_2d_dataset = NoTears(config,human36mGT)

    elif source == "Human36mGT2D":
        poseset = ""
        input_2d_dataset = Human36mGT2D(config,human36mGT)

    elif source == "CPN":
        poseset = "2D"
        input_2d_dataset = CPN2D(config,human36mGT)

    subjects_conf = [a.strip() for a in config["exec"]["Subjects"].split(",")]
    subjects = input_2d_dataset.subject_list
    subjects = list(set(subjects).intersection(set(subjects_conf)))
    '''
    if source ==  "Human3.6m":
        poseset = "2D_detectron_pt"

        keypoints2D = np.load(config.get(source, poseset), allow_pickle=True)
        keypoints = keypoints2D['positions_2d'].item()
        if "detectron" in poseset: # Use Coco-Annotations
            bonemat = Bones.get("coco").transpose()
            bonemat_nohead = Bones.get("coco",{"noHead":True}).transpose()
        else:
        #                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16
            bonemat = Bones.get("notears").transpose()
            bonemat_nohead = Bones.get("notears",{"noHead":True}).transpose()


        for subject in keypoints.keys():
            for action in keypoints[subject]:
                for cam_idx, kps in enumerate(keypoints[subject][action]):
                    cam = dataset.cameras()[subject][cam_idx]
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                    keypoints[subject][action][cam_idx] = kps

        kp1_ = dict()
        kp2_ = dict()
        gt_ = dict()
        c1 = dict()
        c2 = dict()
        acc1 = dict()
        acc2 = dict()
        acc1mul2 = dict()
        subjects = list(dataset.subjects())

        #####DEBUG####
        #subjects = ["S1",]

        cam1_idx = 0
        cam2_idx = 1
        for s in subjects:
            kp1_[s] = dict()
            kp2_[s] = dict()
            gt_[s] = dict()
            c1[s] = dataset.cameras()[s][cam1_idx]
            c2[s] = dataset.cameras()[s][cam2_idx]
            acc1[s] = dict()
            acc2[s] = dict()
            acc1mul2[s] = dict()

            #c1[s]["center"][1] *= -1
            for k in keypoints[s].keys():
                kp1_[s][k] = np.array(keypoints[s][k][cam1_idx][...,:2])
                kp2_[s][k] = np.array(keypoints[s][k][cam2_idx][...,:2])
                #gt_[s][k] =  np.array(dataset._data[s][k]["positions"])
                try:
                    gt_[s][k] = np.array(dataset._data[s][k]["positions"])
                    if "detectron" in poseset:
                        gt_[s][k][:, 0] = np.nan
                        gt_[s][k] = gt_[s][k][:, (9, 0, 0, 0, 0, 14, 11, 15, 12, 16, 13, 1, 4, 2, 5, 3, 6), :]

                except:
                    print("GT values for {} {} do not exist".format(s, k))
                    gt_[s][k] = None
                acc1[s][k] = np.ones((kp1_[s][k].shape[0],kp1_[s][k].shape[1]))
                acc2[s][k] = np.ones((kp1_[s][k].shape[0],kp1_[s][k].shape[1]))
                acc1mul2[s][k] = np.array(acc1[s][k]) * np.array(acc2[s][k])

                #kp1_[s][k][:,:,1] *= -1
                #kp2_[s][k][:,:,1] *= -1


    '''

    #for k in  keypoints["S1"].keys():
    #    estimateForKeypoint(k)

    estimatorInput = list()
    #subjects = ["S8",]
    for s in sorted(subjects):
        actionlist = input_2d_dataset.action_list(s)
        if "Actions" in config["exec"].keys():
            actions_conf = [a.strip() for a in config["exec"]["Actions"].split(",")]
            actionlist = list(set(actionlist).intersection(set(actions_conf)))
        for k in sorted(actionlist):
            #estimatorInput.append([kp1_[s][k], kp2_[s][k], gt_[s][k], c1[s], c2[s], k, s, acc1mul2[s][k],acc1[s][k],acc2[s][k]])
            estimatorInput.append([s,k,input_2d_dataset,cam1_idx,cam2_idx,human36mGT,outpath])

    #estimatorInput.sort(key=lambda x : x[0].shape[0], reverse=True)
    ################
    ### MAINLOOP ###

    if (config["exec"].getint("threads")>1): #multiprocessing
        raise NotImplementedError()
        #with Pool(config["exec"].getint("threads")) as p:
        #    estimatorOutput = p.map(estimateForKeypoint,estimatorInput,chunksize=1)

    else:
        estimatorOutput = list()
        for i in range(config.getint("exec", "repeat")):
            for ei in estimatorInput:
                if config.getint("exec","repeat") > 1:
                    fa = "_Run{}".format(i+1)
                else:
                    fa = ""
                estimatorOutput.append(estimateForKeypoint(ei,iterellipse = config.getint("exec","iterEllipse"),iterbone = config.getint("exec","iterBone"),config = config,filenameappend=fa))

    print("Save to File")
    e3d = Estimated3dDataset(estimatorInput, estimatorOutput)
    np.savez_compressed(os.path.join(outpath,"3D-Estimation_" + source + "_" + poseset), positions_3d=e3d._data)
    print("Saved")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    main(sys.argv)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
