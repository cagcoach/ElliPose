# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import random
import sys, os
#from multiprocessing.pool  import ThreadPool as Pool
from pathos.multiprocessing import ProcessingPool as Pool

import cvxpy
import numpy as np
import scipy
import scipy.linalg
import scipy.ndimage
import matplotlib.pyplot
import cvxpy as cp
import torch

from scipy.spatial.transform import Rotation

from external.VideoPose3D.common.visualization import render_animation
from src.Estimated3dDataset import Estimated3dDataset
from src.h36m_noTears import Human36mNoTears
import traceback


sys.path.append(os.path.abspath('external/VideoPose3D'))
from external.VideoPose3D.common.camera import normalize_screen_coordinates, camera_to_world
from external.VideoPose3D.common.loss import mpjpe, n_mpjpe, p_mpjpe
from external.VideoPose3D.common.h36m_dataset import Human36mDataset
from scipy import stats
import cv2
import cvxpy as cp


from configparser import ConfigParser


def bonelength(poses, bonemat=None):
    '''
    HIP = 0
    R_HIP = 1
    R_KNEE = 2
    R_FOOT = 3
    L_HIP = 4
    L_KNEE = 5
    L_FOOT = 6
    SPINE = 7
    THORAX = 8
    NOSE = 9
    HEAD = 10
    L_SHOULDER = 11
    L_ELBOW = 12
    L_WRIST = 13
    R_SHOULDER = 14
    R_ELBOW = 15
    R_WRIST = 16
    '''
    if bonemat is None:
    #                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16
        raise ValueError("Bonemat is None")

    nanvalues = np.array(np.where(np.isnan(poses))).transpose()
    for numb in range(len(nanvalues)):
        i = nanvalues[numb]
        poses[i[0],i[1],i[2]] = numb * 1e20

    dimensiondiff = (np.moveaxis(poses, 1, 2).reshape(-1,17) @ bonemat).reshape(-1,3,bonemat.shape[1])
    dimensiondiff = np.moveaxis(dimensiondiff, 1,2)

    dimensiondiff[dimensiondiff<=-9e19] = np.nan
    dimensiondiff[dimensiondiff>=9e19] = np.nan

    length = np.sqrt(np.sum(np.square(dimensiondiff), axis=2))

    return length, bonemat

def align(predicted, target):
    """
    rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.nanmean(target, axis=1, keepdims=True)
    muY = np.nanmean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.nansum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.nansum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return predicted_aligned

def correctDistortion(newprediction, bonemat):


    medianpt = np.nanmedian(newprediction.reshape(-1,3),axis=0)
    bonevect = np.array([newprediction[..., 0] @ bonemat, newprediction[..., 1] @ bonemat,
                         newprediction[..., 2] @ bonemat])
    meanbonelen = np.nanmean(np.linalg.norm(bonevect, axis=0), axis=0)

    print("OLDMEAN: {}".format(np.nanmean(np.linalg.norm(bonevect, axis=0), axis=0)))
    print("OLDVAR: {}".format(np.nanvar(np.linalg.norm(bonevect, axis=0), axis=0)))



    # bonevect = bonevect/meanbonelen[None,None,:]
    #boneid = np.array(range(len(meanbonelen))).astype(float)
    #newbonevect = np.empty((4, bonevect.shape[1], bonevect.shape[2]))
    ##newbonevect[:3] = bonevect
    #newbonevect[3] = boneid[None, :]
    newbonevect = bonevect
    #bonevect = newbonevect.reshape(4, -1)
    #bonevect = bonevect[:, ~np.isnan(bonevect).any(axis=0)]

    Data = np.empty((newbonevect.shape[2],newbonevect.shape[1], 5 + len(meanbonelen)))

    Data[:, :, :5] = np.array([np.square(newbonevect[0]) + np.square(newbonevect[1]) - 2 * np.square(newbonevect[2]),
                            np.square(newbonevect[0]) - 2 * np.square(newbonevect[1]) + np.square(newbonevect[2]),
                            4 * newbonevect[0] * newbonevect[1],
                            2 * newbonevect[0] * newbonevect[2],
                            2 * newbonevect[1] * newbonevect[2],
                            ]).transpose()
    Data[:, :, 5:] = 0
    for i in range(bonemat.shape[1]):
        Data[i,:,5+i] = 1

    e = np.array([np.square(bonevect[0]) + np.square(bonevect[1]) + np.square(bonevect[2])]).transpose()

    samplesize =  10 # plus one of each bone for scale
    RansacSteps = 1000
    sampleRanges = [list(np.array(range(k.shape[0]))[~np.isnan(k).any(axis=1)]) for k in bonevect.swapaxes(0,2)]
    #inliersThreshold = 0.0005
    inliersThreshold = 0.003



    samples = list()
    for _ in range(RansacSteps*40):
        choices = np.array(random.choices(range(Data.shape[0]), k=samplesize))
        samples.append ([random.sample(sampleRanges[i], (choices==i).sum()+1) for i in range(bonevect.shape[2])])

    DataParam = cp.Parameter((Data.shape[0] + samplesize,Data.shape[2]))
    eParam = cp.Parameter((Data.shape[0] + samplesize,1))

    s = cp.Variable((5 + len(meanbonelen), 1))
    obj_fun = cvxpy.sum(cvxpy.abs(DataParam @ s - eParam))
    #constraint = [cp.sum(s[5:]) == bonemat.shape[1],]
    constraint = []
    objective = cvxpy.Minimize(obj_fun)
    prob = cvxpy.Problem(objective=objective, constraints=constraint)

    ransacInliers = list()
    countOverTreshold = 0

    for i in range(RansacSteps*40):

        sample = samples[i]
        DataList = list()
        eList = list()
        for a in range(len(sample)):
            DataList.extend(Data[a,sample[a]])
            eList.extend(e[a,sample[a]])
        DataParam.value = np.array(DataList)
        eParam.value = np.array(eList)

        result = prob.solve(verbose=False)

        AbsError = np.abs(Data.reshape(-1,Data.shape[2]) @ s.value - e.reshape(-1,1)).squeeze()
        AbsError = AbsError.reshape(Data.shape[0],Data.shape[1])
        AbsError /= np.sqrt(s.value[5:])
        AbsError = AbsError.reshape(-1)


        #fittness = 1-(AbsError/inliersThreshold)
        #fittness[fittness<0] = 0

        ransacInliers.append((AbsError < inliersThreshold))
        #ransacInliers.append(fittness)
        if (ransacInliers[-1].sum() > (ransacInliers[-1].shape[0] * 0.5)):
            countOverTreshold += 1
            if countOverTreshold >= RansacSteps:
                break
    print("{} samples analyzed, {} good samples found.".format(len(ransacInliers),countOverTreshold))

        #print(np.nanmax(AbsError), np.nanmedian(AbsError),ransacInliers[-1])


    amax = np.argmax(np.sum(np.array(ransacInliers),axis=1))
    data_ = Data.reshape(-1,Data.shape[2])[ransacInliers[amax]>1e-6]
    e_ = e.reshape(-1,1)[ransacInliers[amax]>1e-6]
    result = cp.Problem(cvxpy.Minimize(cvxpy.sum(cvxpy.abs(data_ @ s - e_))),constraints=constraint).solve()

    #sample ellipse


    s = s.value.squeeze()
    Aa = (1 - s[0] - s[1]) / 3
    Ab = Aa + s[1]
    Ac = Aa + s[0]
    Ad = -4 * s[2] / 6
    Ae = -2 * s[3] / 6
    Af = -2 * s[4] / 6

    A = np.array([[Aa, Ad, Ae],
                  [Ad, Ab, Af],
                  [Ae, Af, Ac]])

    A_ = np.linalg.eig(A)

    '''
    randomData = np.random.random((2, 2000)) * 8 - 4

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
    '''

    npr = newprediction.reshape(-1, 3)
    # npr = A_[1] @ npr.transpose()
    npr = np.linalg.inv(A_[1]) @ npr.transpose()
    sv = np.sqrt(np.abs(A_[0][:, None]))
    sv /= np.mean(sv)
    npr *= sv
    npr = A_[1] @ npr
    # npr = np.linalg.inv(A_[1]) @ npr
    npr_ = npr.transpose().reshape(newprediction.shape)
    bonevect_ = np.array([npr_[..., 0] @ bonemat, npr_[..., 1] @ bonemat,
                          npr_[..., 2] @ bonemat])
    print("NEWMEAN: {}".format(np.nanmean(np.linalg.norm(bonevect_, axis=0), axis=0)))
    print("NEWVAR: {}".format(np.nanvar(np.linalg.norm(bonevect_, axis=0), axis=0)))

    new_medianpt = np.nanmedian(npr_.reshape(-1, 3), axis=0)
    npr_ += (medianpt - new_medianpt)[None,None,:]
    #npr_ -= new_medianpt[None,None,:]

    '''
    matplotlib.use("TKAgg")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d([-0.2, 0.2])
    ax.set_ylim3d([-0.2, 0.2])
    ax.set_zlim3d([-0.2, 0.2])
    x = bonevect[0,bonevect[3,:]==2]
    y = bonevect[1,bonevect[3,:]==2]
    z = bonevect[2,bonevect[3,:]==2]
    x_ = bonevect_[0,:,2]
    y_ = bonevect_[1,:,2]
    z_ = bonevect_[2,:,2]
    x__ = ellipse[0]*0.1
    y__ = ellipse[1]*0.1
    z__ = ellipse[2]*0.1
    data, = ax.plot(x,y,z, linestyle="", marker="o", color="blue")
    data2, = ax.plot(x_,y_,z_, linestyle="", marker = "o", color="red")
    data3, = ax.plot(x__,y__,z__, linestyle="", marker = "o", color="yellow")
    plt.show(block=False)
    fig.canvas.draw()
    '''

    return npr_



def points2Dto3D(points1, points2):
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

def optimize2DPointLocationUsingMeanBoneLength(P : np.array, P_: np.array, prediction, bestEstimate = None, bonemat=None, x1 = None, x2 = None, acc1 = None, acc2=None):
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

    bl, bonemat = bonelength(prediction, bonemat)


    meanbl = np.mean(bl, axis=0)

    target_bone = (np.moveaxis(bestEstimate, 1, 2).reshape(-1, 17) @ bonemat).reshape(-1, 3, bonemat.shape[1])
    target_bone *= (meanbl/np.linalg.norm(target_bone,axis=1))[:,None,:]
    target_bone = np.moveaxis(target_bone, 1, 2)

    bones_ = np.array([np.where(bonemat.transpose()==1)[1],np.where(bonemat.transpose()==-1)[1]]).transpose()
    bones = cp.Parameter(bones_.shape)
    #acc1_ = cp.Parameter(17, nonneg=True)
    #acc2_ = cp.Parameter(17, nonneg=True)
    bones.value = bones_

    np.where(np.any(bonemat==1, axis=1))

    target_bone = np.append(target_bone,np.ones(target_bone.shape-np.array([0,0,2])),axis=2)

    prediction = np.append(prediction,np.ones(prediction.shape-np.array([0,0,2])),axis=2)
    newprediction = np.copy(prediction)

    tb = cp.Parameter(target_bone[0].shape)
    Px = cp.Parameter((17,3))
    P_x = cp.Parameter((17,3))
    x_ = cp.Variable((17,3))
    X = cp.Parameter((17,3))


    def globalDist(x, x_):
        return (x[0] - x_[0]) ** 2 + (x[1] - x_[1]) ** 2 + (x[2] - x_[2]) ** 2

    def projdist(p1, p_1,x_):
        p2 = P[:,:3]@x_ + P[:,3]
        p2 *= p1[2] #p1[2] is already weighted by acc1 thus no weight needed

        p_2 = P_[:,:3]@x_ + P_[:,3]
        p_2 *= p_1[2] #p_1[2] is already weighted by acc2 thus no weight needed

        #return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p_1[0]-p_2[0])**2 + (p_1[1]-p_2[1])**2
        return (p1[0] - p2[0]) ** 2 \
               + (p1[1] - p2[1]) ** 2 \
               + (p_1[0] - p_2[0]) ** 2 \
               + (p_1[1] - p_2[1]) ** 2


    def bonediff(bone,x,tbij):
        xbj0 = x[bone[0].value]
        xbj1 = x[bone[1].value]
        return ((xbj0[0] - xbj1[0] - tbij[0]) ** 2
               +(xbj0[1] - xbj1[1] - tbij[1]) ** 2
               +(xbj0[2] - xbj1[2] - tbij[2]) ** 2)

    objarray = list()

    term = (cp.sum([projdist(Px[k], P_x[k], x_[k]) for k in range(17)]) +
            #cp.sum([globalDist(X[k], x_[k]) for k in range(17)]) +
            20 * cp.sum([bonediff(bones[k],x_, tb[k]) for k in range(bones.shape[0])]))

    obj = cp.Minimize(term)
    prob = cp.Problem(obj)

    frames = prediction.shape[0]

    #for i in range(0,frames):

    def process(i):


        p1 = (P[:,:3] @ bestEstimate[i].transpose() + P[:,3, None]).transpose()
        if x1 is None:
            p1[:, :2] /= p1[:, 2, None]
        else:
            p1[:, :2] = x1[i]
        p1[:, 2] = 1 / p1[:, 2] # invert 3rd row prevend division
        p1 *= acc1[i,:,None] / (acc1[i,:,None] + acc2[i,:,None]) # already weight point
            # here p1 is weighted by the prediction accuracy. Since we also weight p1[2] which is later
            # multiplied to the point we dont need to multiply acc1 later explicitly to the predicted point

        Px.value = p1

        p_1 = (P_[:,:3] @ bestEstimate[i].transpose() + P[:,3, None]).transpose()
        if x2 is None:
            p_1[:, :2] /= p_1[:, 2, None]
        else:
            p_1[:, :2] = x2[i]
        p_1[:, 2] = 1 / p_1[:, 2] # invert 3rd row prevend division
        p_1 *= acc2[i,:,None]  / (acc1[i,:,None] + acc2[i,:,None])  # already weight point
            # here p_1 is weighted by the prediction accuracy. Since we also weight p_1[2] which is later
            # multiplied to the point we dont need to multiply acc2 later explicitly to the predicted point
        P_x.value = p_1

        tb.value = target_bone[i]

        x_.value = bestEstimate[i,:,:3]
        X.value = bestEstimate[i,:,:3]
        prob.solve()
        #if (i % 10 == 0): print(i, "/", prediction.shape[0], end="\r")
        return x_.value
        #

    for i in range(frames):
        try:
            newprediction[i,:,:3] = process(i)
        except ValueError:
            #print("ValueError. Continue")
            #print(traceback.format_exc())
            newprediction[i,:,:3] = np.nan

    #with Pool(processes=10) as pool:
    #    newprediction[:,:,:3] = np.array(pool.map(process, range(frames)))
    #obj = cp.Minimize(cp.sum(objarray))
    #prob = cp.Problem(obj)
    #prob.solve(verbose=True)

    retarray = np.empty([len(mask),prediction.shape[1],3])
    retarray[mask == False] = np.nan
    retarray[mask == True] = newprediction[:,:,:3]
    return retarray


def optimizePUsingMeanBoneLength(P,P_,prediction,bestEstimate = None, x1 = None, x2 = None):
    if bestEstimate is None:
        bestEstimate = prediction


    mask = (np.isnan(bestEstimate).any(axis=2).any(axis=1) == False)
    bestEstimate = bestEstimate[mask]
    prediction = prediction[mask]

    estimationRange = list(range(prediction.shape[0]))
    estimationRange = random.choices(estimationRange,k=100)

    prediction = np.append(prediction,np.ones(prediction.shape-np.array([0,0,2])),axis=2)

    Px =  [cp.Parameter((17,3)),] * len(estimationRange)
    X =  [cp.Parameter((17,3)),] * len(estimationRange)


    #Rquart = cp.Parameter(4,value=[0,0,0,0])

    #w = Rquart[0]
    #x = Rquart[1]
    #y = Rquart[2]
    #z = Rquart[3]

    #R = cp.vstack([cp.hstack([1 - 2*y*y - 2*z*z ,     2*x*y - 2*z*w,     2*x*z + 2*y*w]),
    #                cp.hstack([    2*x*y + 2*z*w , 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w]),
    #                cp.hstack([    2*x*z - 2*y*w ,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y])])#


    R = cp.Parameter((3,3), value = [[1,0,0],[0,1,0],[0,0,1]])
    T = cp.Variable(3, value=[0, 0, 0])

    def projdist(p1, x_):
        #p2 = P[:,:3]@R@x_ + P[:,3] + T
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

            X[i].value = bestEstimate[estimationRange[i],:,:3]

        prob.solve()

        P__[:, :3] = P__[:, :3] @ R.value
        P__[:, 3] = P__[:, 3] + T.value

        return P__

    newP = process(P, x1)
    newP_ = process(P_, x2)

    return newP, newP_

def main(argv):
    config = ConfigParser()
    if len(argv) != 2:
        print("Wrong number of arguments")
        exit()
    config.read(sys.argv[1])

    source = "Human3.6m"
    #source = "NoTears"

    dataset = Human36mDataset(config.get(source, "3D"))


    if source == "NoTears":

        poseset = "2D"
                    #0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16
        bonemat = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0],  # hip
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,-1, 0, 0, 0],  # left upper leg
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,-1, 0],  # left lower leg
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,-1, 0, 0],  # right upper leg
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,-1],  # right lower leg
                            [0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # shoulder
                            [0, 0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # left upper arm
                            [0, 0, 0, 0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0],  # left lower arm
                            [0, 0, 0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0],  # right upper arm
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0, 0, 0],  # right lower arm
                            [1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # nose-lefteye
                            [1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # nose-righteye
                            [1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # nose-leftear
                            [1, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # nose-rightear
                            [0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # lefteye-righteye
                            [0, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # lefteye-leftear
                            [0, 1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # lefteye-rightear
                            [0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # righteye-leftear
                            [0, 0, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # righteye-rightear
                            [0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # leftear-rightear
                            ]).transpose()

        bonemat_nohead = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],  # hip
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0],  # left upper leg
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0],  # left lower leg
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0],  # right upper leg
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1],  # right lower leg
                            [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # shoulder
                            [0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # left upper arm
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0],  # left lower arm
                            [0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # right upper arm
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0],  # right lower arm
                            #[1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # nose-lefteye
                            #[1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # nose-righteye
                            #[1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # nose-leftear
                            #[1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # nose-rightear
                            #[0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # lefteye-righteye
                            #[0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # lefteye-leftear
                            #[0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # lefteye-rightear
                            #[0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # righteye-leftear
                            #[0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # righteye-rightear
                            #[0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # leftear-rightear
                            ]).transpose()

        keypoints = Human36mNoTears(config.get(source, poseset))._data
        accuracies = dict()
        for subject in keypoints.keys():
            for action in keypoints[subject]:
                for cam_idx, kps in enumerate(keypoints[subject][action]["positions"]):
                    if kps is None:
                        continue
                    cam = dataset.cameras()[subject][cam_idx]
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                    keypoints[subject][action][cam_idx] = kps
                if not subject in accuracies:
                    accuracies[subject] = dict()
                if not action in accuracies[subject]:
                    accuracies[subject][action] = dict()

                for cam_idx, acc in enumerate(keypoints[subject][action]["accuracy"]):
                    if acc is None:
                        continue
                    accuracies[subject][action][cam_idx] = acc

        kp1_ = dict()
        kp2_ = dict()
        gt_ = dict()
        c1 = dict()
        c2 = dict()
        acc1 = dict()
        acc2 = dict()
        acc1mul2 = dict()
        subjects = list(keypoints.keys())

        #####DEBUG####
        #subjects = ["S1",]

        cam1_idx = 0
        cam2_idx = 1

        for s in subjects:
            kp1_[s] = dict()
            kp2_[s] = dict()
            gt_[s] = dict()
            acc1[s] = dict()
            acc2[s] = dict()
            acc1mul2[s] = dict()
            c1[s] = dataset.cameras()[s][cam1_idx]
            c2[s] = dataset.cameras()[s][cam2_idx]

            # c1[s]["center"][1] *= -1

            for k in keypoints[s].keys():
                #copy if None prediction

                if cam1_idx in keypoints[s][k] and cam2_idx in keypoints[s][k]:
                    kp1_[s][k] = np.array(keypoints[s][k][cam1_idx])
                    kp2_[s][k] = np.array(keypoints[s][k][cam2_idx])
                    acc1[s][k] = np.array(accuracies[s][k][cam1_idx])
                    acc2[s][k] = np.array(accuracies[s][k][cam2_idx])
                    acc1mul2[s][k] = np.array(accuracies[s][k][cam1_idx]) * np.array(accuracies[s][k][cam2_idx])
                    try:
                        gt_[s][k] = np.array(dataset._data[s][k]["positions"])
                        gt_[s][k][:,0] = np.nan
                        gt_[s][k] = gt_[s][k][:,(9,0,0,0,0,14,11,15,12,16,13,1,4,2,5,3,6),:]
                    except:
                        print("GT values for {} {} do not exist".format(s,k))
                        gt_[s][k] = None

                    # kp1_[s][k][:,:,1] *= -1
                    # kp2_[s][k][:,:,1] *= -1

    if source ==  "Human3.6m":


        poseset = "2D_cpn"

        keypoints2D = np.load(config.get(source, poseset), allow_pickle=True)
        keypoints = keypoints2D['positions_2d'].item()

        #                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16
        bonemat = np.array([[0, 1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # hip
                            [0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # left upper leg
                            [0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # left lower leg
                            [0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # right upper leg
                            [0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # right lower leg
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,-1, 0, 0],  # shoulder
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0],  # left upper arm
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 0, 0, 0],  # left lower arm
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 0],  # right upper arm
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1],  # right lower arm
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0],  # nose-tophead
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,-1, 0, 0, 0, 0, 0],  # shoulder mid-left
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,-1, 0, 0],  # shoulder mid-right
                            [1, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # hip mid-left
                            [1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # hip mid-right
                            ]).transpose()

        bonemat_nohead = np.array([[0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # hip
                            [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # left upper leg
                            [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # left lower leg
                            [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # right upper leg
                            [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # right lower leg
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0],  # shoulder
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],  # left upper arm
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],  # left lower arm
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],  # right upper arm
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],  # right lower arm
                            #[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],  # nose-tophead
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0],  # shoulder mid-left
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0],  # shoulder mid-right
                            [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # hip mid-left
                            [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # hip mid-right
                            ]).transpose()


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
                kp1_[s][k] = np.array(keypoints[s][k][cam1_idx])
                kp2_[s][k] = np.array(keypoints[s][k][cam2_idx])
                gt_[s][k] =  np.array(dataset._data[s][k]["positions"])
                acc1[s][k] = np.ones((kp1_[s][k].shape[0],kp1_[s][k].shape[1]))
                acc2[s][k] = np.ones((kp1_[s][k].shape[0],kp1_[s][k].shape[1]))
                acc1mul2[s][k] = np.array(acc1[s][k]) * np.array(acc2[s][k])

                #kp1_[s][k][:,:,1] *= -1
                #kp2_[s][k][:,:,1] *= -1

    def estimateForKeypoint(var):


        kp1 = var[0]
        kp2 = var[1]
        gt = var[2]
        c1 = var[3]
        c2 = var[4]
        k = var[5]
        s = var[6]

        ###### DEBUG ######
        #if k != "Sitting 1" or s != "S1":
        #    return None

        #open("{}_{}.running".format(k,s), 'a').close()

        acc1mul2 = var[7]
        acc1 = var[8]
        acc2 = var[9]
        #kp1 = keypoints["S1"][k][0]
        #kp2 = keypoints["S1"][k][1]
        #gt = dataset._data["S1"][k]["positions"]

        #kp1 = kp1_[k]
        #kp2 = kp2_[k]
        #gt = gt_[k]

        m = min(kp1.shape[0], kp2.shape[0], gt.shape[0])

        kp1 = kp1[:m,:,:]
        kp2 = kp2[:m,:,:]
        gt = gt[:m,:,:]
        acc1mul2 = acc1mul2[:m,:]
        acc1 = acc1[:m, :]
        acc2 = acc2[:m, :]


        blur = False

        if blur:
            kp1_ = scipy.ndimage.gaussian_filter1d(kp1, 1.5,axis=0)
            kp2_ = scipy.ndimage.gaussian_filter1d(kp2, 1.5, axis=0)

            kp1[np.isnan(kp1_) == False] = kp1_[np.isnan(kp1_) == False]
            kp2[np.isnan(kp2_) == False] = kp2_[np.isnan(kp2_) == False]

        kp1_ = kp1.reshape(-1, 2)
        kp2_ = kp2.reshape(-1, 2)

        #prediction = points2Dto3D(kp1, kp2)[:,:3].reshape(-1, 17, 3)

        #using fundmat
        '''
        fundmat = cv2.findFundamentalMat(kp1, kp2, cv2.FM_RANSAC, ransacReprojThreshold=0.0001,
                                         confidence=0.999)

        P = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]])

        # e = scipy.linalg.null_space(fundmat[0])
        def svdsolve(a, b):
            import numpy as np
            u, s, v = np.linalg.svd(a)
            c = np.dot(u.T, b)
            w = np.linalg.solve(np.diag(s), c)
            x = np.dot(v.T, w)
            return x

        e_ = scipy.linalg.solve(fundmat[0],np.zeros((3,1)))
        e_ = svdsolve(fundmat[0],np.zeros((3,1)))
        U, s, Vh = scipy.linalg.svd(fundmat[0])
        P_ = np.append(np.cross(e_.squeeze(), fundmat[0]), e_, axis=1)

        prediction = cv2.triangulatePoints(P, P_, kp1.transpose(), kp2.transpose()).transpose()
        prediction /= prediction[:,3]
        prediction = prediction[:,:3].reshape(-1, 17, 3)
        '''
        #kp1[12][1] = 11237. # Artificial outlier
        #kp1[12][0] = 11237.

        #c1 = dataset.cameras()["S1"][0]
        #c2 = dataset.cameras()["S1"][1]

        cameraMatrix1 = np.array( [[c1["focal_length"][0],                      0, c1["center"][0]],
                                   [                    0,  c1["focal_length"][1], c1["center"][1]],
                                   [                    0,                      0,               1]])

        cameraMatrix2 = np.array([[c2["focal_length"][0], 0, c2["center"][0]],
                                  [0, c2["focal_length"][1], c2["center"][1]],
                                  [0, 0, 1]])


        goodPredictionMask = (acc1mul2>=np.nanmedian(acc1mul2)).reshape(-1)
        try:
            E=cv2.findEssentialMat(kp1_[goodPredictionMask], kp2_[goodPredictionMask], cameraMatrix1=cameraMatrix1, cameraMatrix2=cameraMatrix2, distCoeffs1=None,
                                distCoeffs2=None, method=cv2.RANSAC, prob=0.999, threshold=0.005)
        except cv2.error:
            traceback.print_exc()
            print("cverror")
        #E = cv2.findEssentialMat(kp1, kp2)
        print(s,k,E[0])
        #fundmat = cv2.findFundamentalMat(kp1,kp2,cv2.FM_RANSAC, ransacReprojThreshold=0.0001, confidence=0.999)

        kp1__ = (kp1_[goodPredictionMask] - cameraMatrix1[None,:2,2]) / cameraMatrix1.diagonal()[None,:2]
        kp2__ = (kp2_[goodPredictionMask] - cameraMatrix2[None,:2,2]) / cameraMatrix2.diagonal()[None,:2]
        retval, R, t, mask, _ = cv2.recoverPose(E[0], kp1__, kp2__, cameraMatrix=np.eye(3), distanceThresh=1000);

        P = np.append(cameraMatrix1, np.array([[0], [0], [0]]), axis=1)

        P_= cameraMatrix2 @ np.append(R,t,axis=1)

        triangulatedPoints = cv2.triangulatePoints(P,P_,kp1_.transpose(),kp2_.transpose())
        triangulatedPoints /= triangulatedPoints[3, :]

        realCameraDist = np.linalg.norm(c1["translation"]-c2["translation"])

        prediction = triangulatedPoints.transpose()[:, :3].reshape(-1, 17, 3)

        #prediction = (prediction @ realCamera1Rot * realCameraDist/np.linalg.norm(t)) + realCamera1Trans

       # print("P-MPJPE: {} {:.2f} mm ({} frames)".format(k,
       #                                                  p_mpjpe((prediction[:, :, :].copy()) * np.array([1, 1, 1]),
       #                                                          (gt[:, :, :])) * 1000, gt.shape[0]))
        newprediction = np.copy(prediction)

        for i in range(5):
            #newprediction = optimize2DPointLocationUsingMeanBoneLength(P, P_, prediction, bestEstimate=newprediction,
            #                                                           bonemat=bonemat, x1=kp1, x2=kp2, acc1=acc1,
            #                                                           acc2=acc2)
            pass

        for i in range(1):
            npr = correctDistortion(newprediction, bonemat_nohead)
            #npr = newprediction
            #npr = npr.transpose()
            #mask = ~np.isnan(npr).any(axis=1)

            '''
            calib_retval, calib_cameraMatrix1, calib_distCoeffs1, calib_cameraMatrix2, calib_distCoeffs2, \
            calib_R, calib_T, calib_E, calib_F  = \
                cv2.stereoCalibrate(objectPoints=npr.reshape(-1,3)[goodPredictionMask][None, ...].astype(np.float32),
                                        imagePoints1=kp1_[goodPredictionMask][None, ...].astype(np.float32),
                                        imagePoints2=kp2_[goodPredictionMask][None, ...].astype(np.float32),
                                        cameraMatrix1=cameraMatrix1.copy(),
                                        distCoeffs1=(0, 0, 0, 0),
                                        cameraMatrix2=cameraMatrix2.copy(),
                                        distCoeffs2=(0, 0, 0, 0),
                                        imageSize=(1, 1),
                                        #R=R.copy(),
                                        #T=t.copy(),
                                        flags=cv2.CALIB_FIX_INTRINSIC,
                                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6))

            scale = cv2.norm(calib_T)
            calib_T /= scale
            npr /= scale

            P = np.append(calib_cameraMatrix1, np.array([[0], [0], [0]]), axis=1)
            P_ = calib_cameraMatrix2 @ np.append(calib_R, calib_T, axis=1)
            '''

            R1 = cv2.solvePnPRansac(objectPoints = npr.reshape(-1,3)[goodPredictionMask],
                                    imagePoints = kp1_[goodPredictionMask],
                                    cameraMatrix = cameraMatrix1,
                                    distCoeffs=(0, 0, 0, 0),
                                    rvec = (cv2.Rodrigues(src=np.eye(3))[0]),
                                    tvec = np.zeros((1,3)),
                                    useExtrinsicGuess=True,
                                    reprojectionError=0.1,
                                    iterationsCount = 10000)

            R2 = cv2.solvePnPRansac(objectPoints=npr.reshape(-1, 3)[goodPredictionMask],
                                    imagePoints=kp2_[goodPredictionMask],
                                    cameraMatrix=cameraMatrix2,
                                    distCoeffs=(0, 0, 0, 0),
                                    rvec=(cv2.Rodrigues(src=R.copy())[0]),
                                    tvec=t.copy(),
                                    useExtrinsicGuess=True,
                                    reprojectionError=0.1,
                                    iterationsCount = 10000)

            shape = npr.shape

            npr = npr.reshape(-1, 3).transpose()
            npr_ = np.empty((4,npr.shape[1]))
            npr_[:3,:] = npr.copy()
            npr_[3,:] = 1

            Rt1 = np.append(np.append(cv2.Rodrigues(R1[1])[0], R1[2], axis=1),[[0,0,0,1]],axis=0)
            Rt2 = np.append(np.append(cv2.Rodrigues(R2[1])[0], R2[2], axis=1),[[0,0,0,1]],axis=0)

            npr_ = Rt1 @ npr_

            Rt2_ = Rt2 @ np.linalg.inv(Rt1)
            Rt1_ = np.eye(4)

            scale = 1/np.linalg.norm(Rt2_[:3,3])

            npr_[:3] *= scale
            Rt2_[:3,3] *= scale

            P =  cameraMatrix1 @ Rt1_[:3]
            P_ = cameraMatrix2 @ Rt2_[:3]
            npr_ = npr_[:3].transpose().reshape(shape)

            #P1 = camera_to_world(Rt1_[:3,3].transpose(), c1["orientation"].astype(float),
            #                     c1["translation"])
            #P2 = camera_to_world(Rt2_[:3,3].transpose(), c1["orientation"].astype(float),
            #                     c1["translation"])

            triangulatedPoints = cv2.triangulatePoints(P, P_, kp1_.transpose(), kp2_.transpose())
            triangulatedPoints /= triangulatedPoints[3, :]
            triangulatedPoints = triangulatedPoints.transpose()[:, :3].reshape(-1, 17, 3)

            newprediction = np.copy(triangulatedPoints)

            #newprediction = npr_

            for i in range(5):
                newprediction = optimize2DPointLocationUsingMeanBoneLength(P, P_, triangulatedPoints, bestEstimate=newprediction,
                                                                           bonemat=bonemat, x1=kp1, x2=kp2, acc1=acc1,
                                                                           acc2=acc2)

        prediction    = camera_to_world((   prediction.reshape(-1,3) * realCameraDist).astype(float),c1["orientation"].astype(float),c1["translation"]).reshape(-1,17,3)
        newprediction = camera_to_world((newprediction.reshape(-1,3) * realCameraDist).astype(float),c1["orientation"].astype(float),c1["translation"]).reshape(-1,17,3)

        #al_prediction = align(prediction,gt)
        #n_prediction = align(newprediction,gt)

        #print("### GT ###")
        gt_length, _ = bonelength(gt, bonemat)
        #print("### Prediction ###")
        pred_length, _ = bonelength(prediction, bonemat)

        new_length, _ = bonelength(newprediction, bonemat)

        np.set_printoptions(precision=3, linewidth = 300)
        print("MEAN:")
        print(np.nanmean(gt_length, axis=0))
        print(np.nanmean(pred_length, axis=0))
        print(np.nanmean(new_length, axis=0))
        print("MEDIAN:")
        print(np.nanmedian(gt_length, axis=0))
        print(np.nanmedian(pred_length, axis=0))
        print(np.nanmedian(new_length, axis=0))

        print("VARIANCE:")
        print(np.nanvar(gt_length, axis=0))
        print(np.nanvar(pred_length, axis=0))
        print(np.nanvar(new_length, axis=0))

        gtmean = np.mean(gt_length, axis=0)

        if(False):
            matplotlib.use("TKAgg")
            fig, ax1 = matplotlib.pyplot.subplots()
            bins = np.linspace(gtmean[9]*0.8, gtmean[9]*1.2, 50)
            #ax1.hist(gt_length[:, 9], bins=bins, alpha=0.5, label='GT')
            ax1.hist(pred_length[:, 9], bins=bins, alpha=0.5, label='PRED')
            ax1.hist(new_length[:, 9], bins=bins, alpha=0.5, label='NEW')


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
        print("P-MPJPE: {} {:.2f} mm ({} frames)".format(k,
            p_mpjpe((prediction[:, :, :]) * np.array([1, 1, 1]), (gt[:, :, :])) * 1000,gt.shape[0]))

        print("new P-MPJPE: {} {:.2f} mm ({} frames)".format(k,
                                                         p_mpjpe((newprediction[:, :, :]) * np.array([1, 1, 1]),
                                                                 (gt[:, :, :])) * 1000, gt.shape[0]))
        #'''

        #os.remove("{}_{}.running".format(k, s))
        return prediction, newprediction, gt, k, s, P, P_
        #break; #FIXME Debug out

    #for k in  keypoints["S1"].keys():
    #    estimateForKeypoint(k)

    estimatorInput = list()
    for s in subjects:
        for k in kp1_[s].keys():
            estimatorInput.append([kp1_[s][k], kp2_[s][k], gt_[s][k], c1[s], c2[s], k, s, acc1mul2[s][k],acc1[s][k],acc2[s][k]])

    estimatorInput.sort(key=lambda x : x[0].shape[0], reverse=True)
    ################
    ### MAINLOOP ###
    if (False): #multiprocessing
        with Pool(10) as p:
            estimatorOutput = p.map(estimateForKeypoint,estimatorInput)
    else:
        estimatorOutput = list()
        for ei in estimatorInput:
            estimatorOutput.append(estimateForKeypoint(ei))


    ################
    #for prediction, newprediction, gt, k, s, P, P_ in estimatorOutput:
    #    print("P-MPJPE: {} {:.2f} mm ({} frames)".format(k,
    #                                                     p_mpjpe((prediction[:, :, :]) * np.array([1, 1, 1]),
    #                                                             (gt[:, :, :])) * 1000, gt.shape[0]))
    #
    #    print("new P-MPJPE: {} {:.2f} mm ({} frames)".format(k,
    #                                                         p_mpjpe((newprediction[:, :, :]) * np.array([1, 1, 1]),
    #                                                                 (gt[:, :, :])) * 1000, gt.shape[0]))

    print("Save to File")
    e3d = Estimated3dDataset(estimatorInput, estimatorOutput, dataset._skeleton)
    np.savez_compressed("3D-Estimation_" + source + "_" + poseset, positions_3d=e3d._data)
    print("Saved")


    '''
    keypoints_metadata = keypoints2D['metadata'].item()
    keypoints_metadata["layout_name"] = ""

    #render_animation(input_keypoints[::4,:,:], keypoints_metadata, {'Reconstruction': prediction[::4,:,(2,0,1)]}, dataset.skeleton(),dataset.fps()/4, 3000,00,"out0_"+poseset+".mp4",viewport=(cam['res_w'], cam['res_h']))
    render_animation(np.zeros(prediction[:-1:4,:,:2].shape),
                     keypoints_metadata,
                     {'Reconstruction': prediction[:-1:4,:,:], 'GT': gt[::4,:,:]},
                     dataset.skeleton(),
                     dataset.fps()/4,
                     3000,
                     0,
                     "out1"+poseset+".mp4",
                     viewport=(cam['res_w'], cam['res_h']))
    '''
    #render_animation(input_keypoints[::4,:,:], keypoints_metadata, {'Reconstruction': prediction[::4,:,(0,1,2)]}, dataset.skeleton(),dataset.fps()/4, 3000,00,"out2"+poseset+".mp4",viewport=(cam['res_w'], cam['res_h']))
    #render_animation(input_keypoints[::4,:,:], keypoints_metadata, {'Reconstruction': prediction[::4,:,(1,2,0)]}, dataset.skeleton(),dataset.fps()/4, 3000,00,"out3"+poseset+".mp4",viewport=(cam['res_w'], cam['res_h']))




    #print(retval)


    #hd = Human36mDataset(path = config.get("Human3.6m","Location"))
    #print(hd._data)



    # Use a breakpoint in the code line below to debug your script.
      # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    main(sys.argv)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
