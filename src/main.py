# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
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


from external.VideoPose3D.common.visualization import render_animation

sys.path.append(os.path.abspath('external/VideoPose3D'))
from external.VideoPose3D.common.camera import normalize_screen_coordinates
from external.VideoPose3D.common.loss import mpjpe, n_mpjpe, p_mpjpe
from external.VideoPose3D.common.h36m_dataset import Human36mDataset
import cv2


from configparser import ConfigParser


def bonelength(poses):
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
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,-1, 0, 0, 0, 0, 0],  #shoulder mid-left
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,-1, 0, 0],  # shoulder mid-right
                        [1, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # hip mid-left
                        [1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # hip mid-right
                        ]).transpose()
    dimensiondiff = (np.moveaxis(poses, 1, 2).reshape(-1,17) @ bonemat).reshape(-1,3,bonemat.shape[1])
    dimensiondiff = np.moveaxis(dimensiondiff, 1,2)

    length = np.sqrt(np.sum(np.square(dimensiondiff),axis=2))


    return length, bonemat

def align(predicted, target):
    """
    rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

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

def points2Dto3D(points1, points2):
    #From Hartley etal. - Miltiple View Geometry in Computer Vision, Page 318

    fundmat = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, ransacReprojThreshold=0.0001, confidence=0.999)

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

def optimizeUsingMeanBoneLength(P,P_,prediction,bestEstimate = None):
    if bestEstimate is None:
        bestEstimate = prediction


    bl, bonemat = bonelength(bestEstimate)
    meanbl = np.mean(bl, axis=0)

    target_bone = (np.moveaxis(bestEstimate, 1, 2).reshape(-1, 17) @ bonemat).reshape(-1, 3, bonemat.shape[1])
    target_bone *= (meanbl/np.linalg.norm(target_bone,axis=1))[:,None,:]
    target_bone = np.moveaxis(target_bone, 1, 2)

    bones_ = np.array([np.where(bonemat.transpose()==1)[1],np.where(bonemat.transpose()==-1)[1]]).transpose()
    bones = cp.Parameter(bones_.shape)
    bones.value = bones_

    np.where(np.any(bonemat==1, axis=1))

    target_bone = np.append(target_bone,np.ones(target_bone.shape-np.array([0,0,2])),axis=2)

    prediction = np.append(prediction,np.ones(prediction.shape-np.array([0,0,2])),axis=2)
    newprediction = np.copy(prediction)

    tb = cp.Parameter(target_bone[0].shape)
    Px = cp.Parameter((17,3))
    P_x = cp.Parameter((17,3))
    x_ = cp.Variable((17,3))

    def projdist(p1, p_1,x_):
        p2 = P[:,:3]@x_ + P[:,3]
        p2 *= p1[2]

        p_2 = P_[:,:3]@x_ + P_[:,3]
        p_2 *= p_1[2]

        #return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p_1[0]-p_2[0])**2 + (p_1[1]-p_2[1])**2
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p_1[0] - p_2[0]) ** 2 + (p_1[1] - p_2[1]) ** 2

    def bonediff(bone,x,tbij):
        xbj0 = x[bone[0].value]
        xbj1 = x[bone[1].value]
        return ((xbj0[0] - xbj1[0] - tbij[0]) ** 2
               +(xbj0[1] - xbj1[1] - tbij[1]) ** 2
               +(xbj0[2] - xbj1[2] - tbij[2]) ** 2)

    objarray = list()

    term = (cp.sum([projdist(Px[k], P_x[k], x_[k]) for k in range(17)]) +
            20*cp.sum([bonediff(bones[k],x_, tb[k]) for k in range(bones.shape[0])]))

    obj = cp.Minimize(term)
    prob = cp.Problem(obj)

    frames = prediction.shape[0]

    #for i in range(0,frames):

    def process(i):


        p1 = (P @ prediction[i].transpose()).transpose()
        p1[:, :2] /= p1[:, 2, None]  # invert 3rd row prevend division
        p1[:, 2] = 1 / p1[:, 2]
        Px.value = p1

        p_1 = (P_ @ prediction[i].transpose()).transpose()
        p_1[:, :2] /= p_1[:, 2, None]  # invert 3rd row prevend division
        p_1[:, 2] = 1 / p_1[:, 2]
        P_x.value = p_1

        tb.value = target_bone[i]

        x_.value = bestEstimate[i,:,:3]

        prob.solve()
        #if (i % 10 == 0): print(i, "/", prediction.shape[0], end="\r")
        return x_.value
        #
        #

    for i in range(frames):
        newprediction[i,:,:3] = process(i)

    #with Pool(processes=10) as pool:
    #    newprediction[:,:,:3] = np.array(pool.map(process, range(frames)))
    #obj = cp.Minimize(cp.sum(objarray))
    #prob = cp.Problem(obj)
    #prob.solve(verbose=True)
    return newprediction[:,:,:3]

def main(argv):
    config = ConfigParser()
    if len(argv) != 2:
        print("Wrong number of arguments")
        exit()
    config.read(sys.argv[1])

    dataset = Human36mDataset(config.get("Human3.6m", "3D"))

    poseset = "2D_cpn"

    keypoints2D = np.load(config.get("Human3.6m", poseset), allow_pickle=True)
    keypoints = keypoints2D['positions_2d'].item()

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    kp1_ = dict()
    kp2_ = dict()
    gt_ = dict()
    for k in keypoints["S1"].keys():
        kp1_[k] = keypoints["S1"][k][0]
        kp2_[k] = keypoints["S1"][k][1]
        gt_[k] = dataset._data["S1"][k]["positions"]
        c1 = dataset.cameras()["S1"][0]
        c2 = dataset.cameras()["S1"][1]

    def estimateForKeypoint(var):
        kp1 = var[0]
        kp2 = var[1]
        gt = var[2]
        c1 = var[3]
        c2 = var[4]
        k = var[5]
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

        kp1 = scipy.ndimage.gaussian_filter1d(kp1, 1.5,axis=0)
        kp2 = scipy.ndimage.gaussian_filter1d(kp2, 1.5, axis=0)

        kp1 = kp1.reshape(-1, 2)
        kp2 = kp2.reshape(-1, 2)

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

        E=cv2.findEssentialMat(kp1, kp2, cameraMatrix1=cameraMatrix1, cameraMatrix2=cameraMatrix2, distCoeffs1=None,
                            distCoeffs2=None, method=cv2.RANSAC, prob=0.5, threshold=0.0001)
        #E = cv2.findEssentialMat(kp1, kp2)

        #fundmat = cv2.findFundamentalMat(kp1,kp2,cv2.FM_RANSAC, ransacReprojThreshold=0.0001, confidence=0.999)


        retval, R, t, mask, triangulatedPoints = cv2.recoverPose(E[0], kp1, kp2, cameraMatrix=cameraMatrix1, distanceThresh=10);

        triangulatedPoints /= triangulatedPoints[3,:]



        P = np.append(cameraMatrix1, np.array([[0], [0], [0]]), axis=1)

        P_= cameraMatrix2 @ np.append(R,t,axis=1)


        prediction = triangulatedPoints.transpose()[:, :3].reshape(-1, 17, 3)
        print("P-MPJPE: {} {:.2f} mm ({} frames)".format(k,
                                                         p_mpjpe((prediction[:, :, :]) * np.array([1, 1, 1]),
                                                                 (gt[:, :, :])) * 1000, gt.shape[0]))
        newprediction = np.copy(prediction)
        for i in range(5):
            newprediction = optimizeUsingMeanBoneLength(P,P_,prediction, bestEstimate=newprediction)
            print("P-MPJPE: {} {:.2f} mm ({} frames)".format(k,
                                                         p_mpjpe((newprediction[:, :, :]) * np.array([1, 1, 1]),
                                                                 (gt[:, :, :])) * 1000, gt.shape[0]))


        al_prediction = align(prediction,gt)
        n_prediction = align(newprediction,gt)


        #print("### GT ###")
        gt_length, _ = bonelength(gt)
        #print("### Prediction ###")
        pred_length, _ = bonelength(al_prediction)

        new_length, _ = bonelength(n_prediction)

        np.set_printoptions(precision=3, linewidth = 300)
        print("MEAN:")
        print(np.mean(gt_length, axis=0))
        print(np.mean(pred_length, axis=0))
        print(np.mean(new_length, axis=0))
        print("MEDIAN:")
        print(np.median(gt_length, axis=0))
        print(np.median(pred_length, axis=0))
        print(np.median(new_length, axis=0))


        print("VARIANCE:")
        print(np.var(gt_length, axis=0))
        print(np.var(pred_length, axis=0))
        print(np.var(new_length, axis=0))


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

        '''
        print("P-MPJPE: {} {:.2f} mm ({} frames)".format(k,
            p_mpjpe((prediction[:, :, :]) * np.array([1, 1, 1]), (gt[:, :, :])) * 1000,gt.shape[0]))

        print("new P-MPJPE: {} {:.2f} mm ({} frames)".format(k,
                                                         p_mpjpe((newprediction[:, :, :]) * np.array([1, 1, 1]),
                                                                 (gt[:, :, :])) * 1000, gt.shape[0]))
        '''
        return prediction, newprediction, gt, k
        #break; #FIXME Debug out

    #for k in  keypoints["S1"].keys():
    #    estimateForKeypoint(k)



    with Pool(20) as p:
        val = [[kp1_[k],kp2_[k],gt_[k],c1,c2,k] for k in keypoints["S1"].keys()]
        ret = p.map(estimateForKeypoint,val)

    for prediction, newprediction, gt, k in ret:
        print("P-MPJPE: {} {:.2f} mm ({} frames)".format(k,
                                                         p_mpjpe((prediction[:, :, :]) * np.array([1, 1, 1]),
                                                                 (gt[:, :, :])) * 1000, gt.shape[0]))

        print("new P-MPJPE: {} {:.2f} mm ({} frames)".format(k,
                                                             p_mpjpe((newprediction[:, :, :]) * np.array([1, 1, 1]),
                                                                     (gt[:, :, :])) * 1000, gt.shape[0]))


    keypoints_metadata = keypoints2D['metadata'].item()
    keypoints_metadata["layout_name"] = ""
    '''
    #render_animation(input_keypoints[::4,:,:], keypoints_metadata, {'Reconstruction': prediction[::4,:,(2,0,1)]}, dataset.skeleton(),dataset.fps()/4, 3000,00,"out0_"+poseset+".mp4",viewport=(cam['res_w'], cam['res_h']))
    render_animation(np.zeros(n_prediction[:-1:4,:,:2].shape),
                     keypoints_metadata,
                     {'Reconstruction': n_prediction[:-1:4,:,:], 'GT': gt[::4,:,:]},
                     dataset.skeleton(),
                     dataset.fps()/4,
                     3000,
                     0,
                     "out1"+poseset+".mp4",
                     viewport=(cam['res_w'], cam['res_h']))
    '''
    #render_animation(input_keypoints[::4,:,:], keypoints_metadata, {'Reconstruction': prediction[::4,:,(0,1,2)]}, dataset.skeleton(),dataset.fps()/4, 3000,00,"out2"+poseset+".mp4",viewport=(cam['res_w'], cam['res_h']))
    #render_animation(input_keypoints[::4,:,:], keypoints_metadata, {'Reconstruction': prediction[::4,:,(1,2,0)]}, dataset.skeleton(),dataset.fps()/4, 3000,00,"out3"+poseset+".mp4",viewport=(cam['res_w'], cam['res_h']))



    kp3d = dataset._data["S1"]["Directions"]["positions"].reshape(-1,3)

    #print(retval)


    #hd = Human36mDataset(path = config.get("Human3.6m","Location"))
    #print(hd._data)



    # Use a breakpoint in the code line below to debug your script.
      # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    main(sys.argv)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
