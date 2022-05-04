import os
from collections import defaultdict, OrderedDict
from configparser import ConfigParser
#from src.Skeleton.Bonemat import Bonemat
import sys

import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib
from cv2 import applyColorMap, COLORMAP_JET as COLORMAP

from external.VideoPose3D.common.camera import normalize_screen_coordinates, camera_to_world
from external.VideoPose3D.common.h36m_dataset import Human36mDataset
from external.VideoPose3D.common.loss import mpjpe, p_mpjpe
from src.DataInterface.Human36mGT import Human36mGT
from src.DataInterface.Human36mGT2D import Human36mGT2D
from src.DataInterface.NoTears import NoTears
from src.DataInterface.PickleFolder import PickleFolder
from src.DataInterface.Sequence import NpDimension
from src.h36m_noTears import Human36mNoTears
from src.Skeleton.Keypoint import Dimension as KpDim, Feature, Keypoint, Position
#from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool

import easygui


def numbercellstyle(val, min_, max_):
    changedval = (val - min_)/(max_-min_) * 127
    bgcolor = applyColorMap(np.array(128 + min(changedval, 127)).astype(np.uint8), COLORMAP)

    textcolor = "fff" if (bgcolor[0,0,2] * 0.2126) + (bgcolor[0,0,1] * 0.7152) + (bgcolor[0,0,0] * 0.0722) < 127 else "000"

    hexbgcolor = bgcolor[0, 0, 0] + 0x100 * bgcolor[0, 0, 1] + 0x10000 * bgcolor[0, 0, 2]
    return "background-color:#{:06x};color:#{};text-align:right".format(hexbgcolor,textcolor)

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
    Hmask = ~np.isnan(H).all(axis=2).all(axis=1)
    U_, s_, Vt_ = np.linalg.svd(H[Hmask])
    U = np.empty(H.shape)
    s = np.empty(H.shape[:2])
    Vt = np.empty(H.shape)
    U[Hmask] = U_
    U[~Hmask] = np.nan
    s[Hmask] = s_
    s[~Hmask] = np.nan
    Vt[Hmask] = Vt_
    Vt[~Hmask] = np.nan

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

def main(conf):
    matplotlib.use("TkAgg")

    config = ConfigParser()

    config.read(conf)


    human36mGT = Human36mGT(config)

    source = config["exec"]["source"]
    refA = config.getfloat("Ellipse","breakingcondition")
    #dataset = Human36mDataset(config.get(source, "3D"))

    '''
    if source ==  "Human3.6m" or source == "Detectron":
    
        poseset = "2D_gt"
    
        keypoints2D = np.load(config.get(source, poseset), allow_pickle=True)
        keypoints = keypoints2D['positions_2d'].item()
    
        #                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16
        bonemat = Bonemat.get("coco").transpose()
    
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
        subjects = list(dataset.subjects())
    
        cam1_idx = 0
        cam2_idx = 1
        for s in subjects:
            kp1_[s] = dict()
            kp2_[s] = dict()
            gt_[s] = dict()
            c1[s] = dataset.cameras()[s][cam1_idx]
            c2[s] = dataset.cameras()[s][cam2_idx]
    
            for k in keypoints[s].keys():
                kp1_[s][k] = np.array(keypoints[s][k][cam1_idx])
                kp2_[s][k] = np.array(keypoints[s][k][cam2_idx])
                gt_[s][k] =  np.array(dataset._data[s][k]["positions"])
    
    '''

    a = PickleFolder(config, "aligned3d")
    htmlhead = "<head><title>{}</title></head>".format(config["exec"]["outpath"])
    htmlbody = "<h1>{}</h1>cameras {} and {}, {} ellipse iteration, {} bonelength iterations, blur: {}".format(config["exec"]["source"], config["exec"]["cameraindex1"], config["exec"]["cameraindex2"], config["exec"]["iterellipse"], config["exec"]["iterbone"], config["exec"]["blur"])
    p_ck30 = dict()
    p_ck50 = dict()
    p_ck100 = dict()
    p_ck250 = dict()

    ck30 = dict()
    ck50 = dict()
    ck100 = dict()
    ck250 = dict()

    #3D-Estimation_Human3.6m_2D_gt
    notears = False
    detectron = True

    outA = ""
    outB = ""
    mpjpehtml = defaultdict(lambda:dict())#"<tr><td>subject</td><td>action</td><td>MPJPE</td><td>PMPJPE</td><td>frames</td></tr>"
    mydata = defaultdict(lambda:defaultdict(lambda:dict()))
    mpjpehtmlhead = "<td>MPJPE</td><td>PMPJPE</td><td>Trajectory</td><td>Best A</td><td>frames</td>"

    keys = a.subject_list
    keys.sort(key=lambda x: int(x[1:]))
    for subject_ in keys:
        p_ck30[subject_] = dict()
        p_ck50[subject_] = dict()
        p_ck100[subject_] = dict()
        p_ck250[subject_] = dict()

        ck30[subject_] = dict()
        ck50[subject_] = dict()
        ck100[subject_] = dict()
        ck250[subject_] = dict()
        mpjpehtml[subject_] = dict()
        mydata[subject_] = defaultdict(lambda:dict())

        items = a.action_list(subject_)
        items.sort(key=lambda x: x)



        #for action in items:
            #for action, prediction in items:
            #gt = gt_[subject_][action]
        def process(action):
            outdict = dict()
            predseq = a.get_sequence(subject_,action)
            bestA = a.get_bestA_for_Sequence(subject_,action)
            predseq.interpolateCenterFromLeftRight(Feature.hip)
            predseq.interpolateCenterFromLeftRight(Feature.shoulder)
            gtseq = human36mGT.get_sequence(subject_,action)
            #
            keypoint_iter = list(set(predseq.npdimensions[NpDimension.KEYPOINT_ITER]).intersection(gtseq.npdimensions[NpDimension.KEYPOINT_ITER]))
            keypoint_iter.sort(key=lambda x: (int(x != Keypoint(Position.center, Feature.hip))))
            frame_iter = list(set(predseq.npdimensions[NpDimension.FRAME_ITER]).intersection(gtseq.npdimensions[NpDimension.FRAME_ITER]))
            frame_iter.sort()


            prediction, f = predseq.get(OrderedDict([(NpDimension.FRAME_ITER,frame_iter),
                                                     (NpDimension.KEYPOINT_ITER,keypoint_iter),
                                                     (NpDimension.KP_DATA, (KpDim.x, KpDim.y, KpDim.z))]))
            gt, f_ = gtseq.get(OrderedDict([(NpDimension.FRAME_ITER, frame_iter),
                                              (NpDimension.KEYPOINT_ITER, keypoint_iter),
                                              (NpDimension.KP_DATA, (KpDim.x, KpDim.y, KpDim.z))]))
            thistrajectory = mpjpe(torch.from_numpy(prediction[:,(0,)]), gt[:,(0,)]) * 1000
            prediction[:,:] -= prediction[:,(0,)]
            gt[:, :] -= gt[:, (0,)]

            thispmpjpe = p_mpjpe( prediction, gt ) * 1000
            thismpjpe =  mpjpe(torch.from_numpy(prediction),gt) * 1000

            #outA += ("P-MPJPE\t{}\t{}\t{:.2f}\tmm\t{}\tframes\n".format(subject_, action, thispmpjpe, gt.shape[0]))
            #outhtmlPMPJPE += (
            #    "<tr><td>{}</td><td>{}</td><td style=\"background-color:#{:06x}\">{:.2f} mm</td><td>{} frames</td></tr>".format(subject_, action, bgcolor, thispmpjpe, gt.shape[0]))

            #outB += ("MPJPE\t{}\t{}\t{:.2f}\tmm\t{}\tframes\n".format(subject_, action, thismpjpe, gt.shape[0]))
            outdict["mpjpehtml"] = "<td style=\"{}\">{:.2f} mm</td><td style=\"{}\">{:.2f} mm</td><td style=\"{}\">{:.2f} mm</td><td style=\"{}\">{:.4f}</td><td>{} frms</td>".format(
                    numbercellstyle(thismpjpe, 0, 100), thismpjpe,numbercellstyle(thispmpjpe, 0, 80), thispmpjpe,numbercellstyle(thistrajectory, 0, 100), thistrajectory, numbercellstyle(np.log10(bestA), np.log10(refA), 1), bestA, gt.shape[0]
                )
            outdict["trajectory"] = thistrajectory
            outdict["mpjpe"] = thismpjpe
            outdict["pmpjpe"] = thispmpjpe
            outdict["bestA"] = bestA
            outdict["frames"] = gt.shape[0]
            allPointsDist = np.linalg.norm(prediction - gt, axis=2).reshape(-1)

            outdict["ck30sum"] =  (allPointsDist <= 0.03).sum()
            outdict["ck50sum"] = (allPointsDist <= 0.05).sum()
            outdict["ck100sum"] = (allPointsDist <= 0.1).sum()
            outdict["ck150sum"] = (allPointsDist <= 0.15).sum()
            outdict["ck250sum"] = (allPointsDist <= 0.25).sum()

            allPointsDist = np.linalg.norm(align(prediction, gt) - gt, axis=2).reshape(-1)

            outdict["p_ck30sum"] = (allPointsDist <= 0.03).sum()
            outdict["p_ck50sum"] = (allPointsDist <= 0.05).sum()
            outdict["p_ck100sum"] = (allPointsDist <= 0.1).sum()
            outdict["p_ck150sum"] = (allPointsDist <= 0.15).sum()
            outdict["p_ck250sum"] = (allPointsDist <= 0.25).sum()
            outdict["totalpoints"] = len(allPointsDist)
            print(subject_ + " " + action)
            return outdict

        with Pool(24) as p:
            mydata[subject_] = defaultdict(lambda: dict(), **dict(zip(items,p.map(process,items))))
        mydata[subject_]["TOTAL"]["trajectory"] = np.mean([mydata[subject_][a]["trajectory"] for a in items])
        mydata[subject_]["TOTAL"]["mpjpe"] = np.mean([mydata[subject_][a]["mpjpe"] for a in items])
        mydata[subject_]["TOTAL"]["pmpjpe"] = np.mean([mydata[subject_][a]["pmpjpe"] for a in items])
    mydata["TOTAL"]["trajectory"] = np.mean([mydata[s]["TOTAL"]["trajectory"] for s in keys])
    mydata["TOTAL"]["mpjpe"] = np.mean([mydata[s]["TOTAL"]["mpjpe"] for s in keys])
    mydata["TOTAL"]["pmpjpe"] = np.mean([mydata[s]["TOTAL"]["pmpjpe"] for s in keys])
    mydata["TOTALVAL"]["trajectory"] = np.mean([mydata[s]["TOTAL"]["trajectory"] for s in set(["S9", "S11"]).intersection(keys)])
    mydata["TOTALVAL"]["mpjpe"] = np.mean([mydata[s]["TOTAL"]["mpjpe"] for s in set(["S9","S11"]).intersection(keys)])
    mydata["TOTALVAL"]["pmpjpe"] = np.mean([mydata[s]["TOTAL"]["pmpjpe"] for s in set(["S9","S11"]).intersection(keys)])

    #htmlbody += "<h2>MPJPE</h2><table>{}</table>".format(outhtml)
    print(outA)
    print("####################################")
    print(outB)
    print("####################################")

    tot = defaultdict(lambda: 0)
    cnt = defaultdict(lambda: 0)
    keys = list(a.subject_list)
    keys.sort(key=lambda x: int(x[1:]))
    pck_table = "<thead style=\"position: sticky; top: 0;\">" \
                "<tr style=\"background-color: #ddd;\">" \
                "<td>subject</td>" \
                "<td>action</td>"\
                +mpjpehtmlhead+\
                "<td>PCK<br />30</td>" \
                "<td>PCK<br />50</td>" \
                "<td>PCK<br />100</td>" \
                "<td>PCK<br />150</td>" \
                "<td>PCK<br />250</td>" \
                "<td>PPCK<br />30</td>" \
                "<td>PPCK<br />50</td>" \
                "<td>PPCK<br />100</td>" \
                "<td>PPCK<br />150</td>" \
                "<td>PPCK<br />250</td>" \
                "</tr>" \
                "</thead>" \
                "<tbody>"
    for subject_ in keys:
        items = list(a.action_list(subject_))
        items.sort(key=lambda x: x)
        for action in items:
            tot["PCK30"] += mydata[subject_][action]["ck30sum"]
            tot["PCK50"] += mydata[subject_][action]["ck50sum"]
            tot["PCK100"] += mydata[subject_][action]["ck100sum"]
            tot["PCK150"] += mydata[subject_][action]["ck150sum"]
            tot["PCK250"] += mydata[subject_][action]["ck250sum"]
            tot["PPCK30"] += mydata[subject_][action]["p_ck30sum"]
            tot["PPCK50"] += mydata[subject_][action]["p_ck50sum"]
            tot["PPCK100"] += mydata[subject_][action]["p_ck100sum"]
            tot["PPCK150"] += mydata[subject_][action]["p_ck150sum"]
            tot["PPCK250"] += mydata[subject_][action]["p_ck250sum"]

            cnt["PCK30"] += mydata[subject_][action]["totalpoints"]
            cnt["PCK50"] += mydata[subject_][action]["totalpoints"]
            cnt["PCK100"] += mydata[subject_][action]["totalpoints"]
            cnt["PCK150"] += mydata[subject_][action]["totalpoints"]
            cnt["PCK250"] += mydata[subject_][action]["totalpoints"]
            cnt["PPCK30"] += mydata[subject_][action]["totalpoints"]
            cnt["PPCK50"] += mydata[subject_][action]["totalpoints"]
            cnt["PPCK100"] += mydata[subject_][action]["totalpoints"]
            cnt["PPCK150"] += mydata[subject_][action]["totalpoints"]
            cnt["PPCK250"] += mydata[subject_][action]["totalpoints"]

            mydata[subject_][action]["pck30"] = mydata[subject_][action]["ck30sum"] / mydata[subject_][action]["totalpoints"]
            mydata[subject_][action]["pck50"] = mydata[subject_][action]["ck50sum"] / mydata[subject_][action]["totalpoints"]
            mydata[subject_][action]["pck100"] = mydata[subject_][action]["ck100sum"] / mydata[subject_][action]["totalpoints"]
            mydata[subject_][action]["pck150"] = mydata[subject_][action]["ck150sum"] / mydata[subject_][action]["totalpoints"]
            mydata[subject_][action]["pck250"] = mydata[subject_][action]["ck250sum"] / mydata[subject_][action]["totalpoints"]
            mydata[subject_][action]["p_pck30"] = mydata[subject_][action]["p_ck30sum"] / mydata[subject_][action]["totalpoints"]
            mydata[subject_][action]["p_pck50"] = mydata[subject_][action]["p_ck50sum"] / mydata[subject_][action]["totalpoints"]
            mydata[subject_][action]["p_pck100"] = mydata[subject_][action]["p_ck100sum"] / mydata[subject_][action]["totalpoints"]
            mydata[subject_][action]["p_pck150"] = mydata[subject_][action]["p_ck150sum"] / mydata[subject_][action]["totalpoints"]
            mydata[subject_][action]["p_pck250"] = mydata[subject_][action]["p_ck250sum"] / mydata[subject_][action]["totalpoints"]

            pck_table += "<tr>" \
                         "<td style=\"font-weight:bold\">{}</td>" \
                         "<td style=\"font-weight:bold\">{}</td>" \
                         "{}" \
                         "<td style=\"{}\">{:.2f}</td>" \
                         "<td style=\"{}\">{:.2f}</td>" \
                         "<td style=\"{}\">{:.2f}</td>" \
                         "<td style=\"{}\">{:.2f}</td>" \
                         "<td style=\"{}\">{:.2f}</td>" \
                         "<td style=\"{}\">{:.2f}</td>" \
                         "<td style=\"{}\">{:.2f}</td>" \
                         "<td style=\"{}\">{:.2f}</td>" \
                         "<td style=\"{}\">{:.2f}</td>" \
                         "<td style=\"{}\">{:.2f}</td>" \
                         "</tr>".format(
                subject_,
                action,
                mydata[subject_][action]["mpjpehtml"],
                numbercellstyle(mydata[subject_][action]["pck30"], 1, 0),
                mydata[subject_][action]["pck30"] * 100,
                numbercellstyle(mydata[subject_][action]["pck50"], 1, 0.2),
                mydata[subject_][action]["pck50"] * 100,
                numbercellstyle(mydata[subject_][action]["pck100"], 1, 0.6),
                mydata[subject_][action]["pck100"] * 100,
                numbercellstyle(mydata[subject_][action]["pck150"], 1, 0.8),
                mydata[subject_][action]["pck150"] * 100,
                numbercellstyle(mydata[subject_][action]["pck250"], 1, 0.9),
                mydata[subject_][action]["pck250"] * 100,
                numbercellstyle(mydata[subject_][action]["p_pck30"], 1, 0),
                mydata[subject_][action]["p_pck30"] * 100,
                numbercellstyle(mydata[subject_][action]["p_pck50"], 1, 0.2),
                mydata[subject_][action]["p_pck50"] * 100,
                numbercellstyle(mydata[subject_][action]["p_pck100"], 1, 0.6),
                mydata[subject_][action]["p_pck100"] * 100,
                numbercellstyle(mydata[subject_][action]["p_pck150"], 1, 0.8),
                mydata[subject_][action]["p_pck150"] * 100,
                numbercellstyle(mydata[subject_][action]["p_pck250"], 1, 0.9),
                mydata[subject_][action]["p_pck250"] * 100
            )
        mydata[subject_]["TOTAL"]["pck30"] = np.mean([mydata[subject_][a]["pck30"] for a in items])
        mydata[subject_]["TOTAL"]["pck50"] = np.mean([mydata[subject_][a]["pck50"] for a in items])
        mydata[subject_]["TOTAL"]["pck100"] = np.mean([mydata[subject_][a]["pck100"] for a in items])
        mydata[subject_]["TOTAL"]["pck150"] = np.mean([mydata[subject_][a]["pck150"] for a in items])
        mydata[subject_]["TOTAL"]["pck250"] = np.mean([mydata[subject_][a]["pck250"] for a in items])
        mydata[subject_]["TOTAL"]["p_pck30"] = np.mean([mydata[subject_][a]["p_pck30"] for a in items])
        mydata[subject_]["TOTAL"]["p_pck50"] = np.mean([mydata[subject_][a]["p_pck50"] for a in items])
        mydata[subject_]["TOTAL"]["p_pck100"] = np.mean([mydata[subject_][a]["p_pck100"] for a in items])
        mydata[subject_]["TOTAL"]["p_pck150"] = np.mean([mydata[subject_][a]["p_pck150"] for a in items])
        mydata[subject_]["TOTAL"]["p_pck250"] = np.mean([mydata[subject_][a]["p_pck250"] for a in items])
    mydata["TOTAL"]["pck30"] = np.mean([mydata[s]["TOTAL"]["pck30"] for s in keys])
    mydata["TOTAL"]["pck50"] = np.mean([mydata[s]["TOTAL"]["pck50"] for s in keys])
    mydata["TOTAL"]["pck100"] = np.mean([mydata[s]["TOTAL"]["pck100"] for s in keys])
    mydata["TOTAL"]["pck150"] = np.mean([mydata[s]["TOTAL"]["pck150"] for s in keys])
    mydata["TOTAL"]["pck250"] = np.mean([mydata[s]["TOTAL"]["pck250"] for s in keys])
    mydata["TOTAL"]["p_pck30"] = np.mean([mydata[s]["TOTAL"]["p_pck30"] for s in keys])
    mydata["TOTAL"]["p_pck50"] = np.mean([mydata[s]["TOTAL"]["p_pck50"] for s in keys])
    mydata["TOTAL"]["p_pck100"] = np.mean([mydata[s]["TOTAL"]["p_pck100"] for s in keys])
    mydata["TOTAL"]["p_pck150"] = np.mean([mydata[s]["TOTAL"]["p_pck150"] for s in keys])
    mydata["TOTAL"]["p_pck250"] = np.mean([mydata[s]["TOTAL"]["p_pck250"] for s in keys])
    mydata["TOTALVAL"]["pck30"] = np.mean([mydata[s]["TOTAL"]["pck30"] for s in set(["S9","S11"]).intersection(keys)])
    mydata["TOTALVAL"]["pck50"] = np.mean([mydata[s]["TOTAL"]["pck50"] for s in set(["S9","S11"]).intersection(keys)])
    mydata["TOTALVAL"]["pck100"] = np.mean([mydata[s]["TOTAL"]["pck100"] for s in set(["S9","S11"]).intersection(keys)])
    mydata["TOTALVAL"]["pck150"] = np.mean([mydata[s]["TOTAL"]["pck150"] for s in set(["S9","S11"]).intersection(keys)])
    mydata["TOTALVAL"]["pck250"] = np.mean([mydata[s]["TOTAL"]["pck250"] for s in set(["S9","S11"]).intersection(keys)])
    mydata["TOTALVAL"]["p_pck30"] = np.mean([mydata[s]["TOTAL"]["p_pck30"] for s in set(["S9","S11"]).intersection(keys)])
    mydata["TOTALVAL"]["p_pck50"] = np.mean([mydata[s]["TOTAL"]["p_pck50"] for s in set(["S9","S11"]).intersection(keys)])
    mydata["TOTALVAL"]["p_pck100"] = np.mean([mydata[s]["TOTAL"]["p_pck100"] for s in set(["S9","S11"]).intersection(keys)])
    mydata["TOTALVAL"]["p_pck150"] = np.mean([mydata[s]["TOTAL"]["p_pck150"] for s in set(["S9", "S11"]).intersection(keys)])
    mydata["TOTALVAL"]["p_pck250"] = np.mean([mydata[s]["TOTAL"]["p_pck250"] for s in set(["S9","S11"]).intersection(keys)])

    for subject_ in keys:
        pck_table += "<tr style=\"background-color:#ddd;\">" \
             "<td style=\"outline: 1px solid #ddd;font-weight:bold\">{}</td>" \
             "<td style=\"outline: 1px solid #ddd;font-weight:bold\">TOTAL</td>" \
             "<td style=\"outline: 1px solid #ddd;{}\">{:.2f} mm</td>" \
             "<td style=\"outline: 1px solid #ddd;{}\">{:.2f} mm</td>" \
             "<td style=\"outline: 1px solid #ddd;{}\">{:.2f} mm</td>" \
             "<td style=\"outline: 1px solid #ddd;{}\">{}</td>" \
             "<td style=\"outline: 1px solid #ddd;{}\">{}</td>" \
             "<td style=\"outline: 1px solid #ddd;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #ddd;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #ddd;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #ddd;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #ddd;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #ddd;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #ddd;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #ddd;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #ddd;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #ddd;{}\">{:.2f}</td>" \
             "</tr>".format(
            subject_,
            numbercellstyle(mydata[subject_]["TOTAL"]["mpjpe"], 0, 100),
            mydata[subject_]["TOTAL"]["mpjpe"],
            numbercellstyle(mydata[subject_]["TOTAL"]["pmpjpe"], 0, 80),
            mydata[subject_]["TOTAL"]["pmpjpe"],
            numbercellstyle(mydata[subject_]["TOTAL"]["trajectory"], 0, 100),
            mydata[subject_]["TOTAL"]["trajectory"],
            "",
            "",
            "",
            "",
            numbercellstyle(mydata[subject_]["TOTAL"]["pck30"], 1, 0),
            mydata[subject_]["TOTAL"]["pck30"] * 100,
            numbercellstyle(mydata[subject_]["TOTAL"]["pck50"], 1, 0.2),
            mydata[subject_]["TOTAL"]["pck50"] * 100,
            numbercellstyle(mydata[subject_]["TOTAL"]["pck100"], 1, 0.6),
            mydata[subject_]["TOTAL"]["pck100"] * 100,
            numbercellstyle(mydata[subject_]["TOTAL"]["pck150"], 1, 0.8),
            mydata[subject_]["TOTAL"]["pck150"] * 100,
            numbercellstyle(mydata[subject_]["TOTAL"]["pck250"], 1, 0.9),
            mydata[subject_]["TOTAL"]["pck250"] * 100,
            numbercellstyle(mydata[subject_]["TOTAL"]["p_pck30"], 1, 0),
            mydata[subject_]["TOTAL"]["p_pck30"] * 100,
            numbercellstyle(mydata[subject_]["TOTAL"]["p_pck50"], 1, 0.2),
            mydata[subject_]["TOTAL"]["p_pck50"] * 100,
            numbercellstyle(mydata[subject_]["TOTAL"]["p_pck100"], 1, 0.6),
            mydata[subject_]["TOTAL"]["p_pck100"] * 100,
            numbercellstyle(mydata[subject_]["TOTAL"]["p_pck150"], 1, 0.8),
            mydata[subject_]["TOTAL"]["p_pck150"] * 100,
            numbercellstyle(mydata[subject_]["TOTAL"]["p_pck250"], 1, 0.9),
            mydata[subject_]["TOTAL"]["p_pck250"] * 100,
        )

    for key, name in {"TOTALVAL":"S9,S11","TOTAL":"ALL"}.items():
        pck_table += "<tr style=\"background-color:#bbb;\">" \
             "<td style=\"outline: 1px solid #bbb;font-weight:bold\">{}</td>" \
             "<td style=\"outline: 1px solid #bbb;font-weight:bold\">TOTAL</td>" \
             "<td style=\"outline: 1px solid #bbb;{}\">{:.2f} mm</td>" \
             "<td style=\"outline: 1px solid #bbb;{}\">{:.2f} mm</td>" \
             "<td style=\"outline: 1px solid #bbb;{}\">{:.2f} mm</td>" \
             "<td style=\"outline: 1px solid #bbb;{}\">{}</td>" \
             "<td style=\"outline: 1px solid #bbb;{}\">{}</td>" \
             "<td style=\"outline: 1px solid #bbb;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #bbb;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #bbb;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #bbb;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #bbb;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #bbb;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #bbb;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #bbb;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #bbb;{}\">{:.2f}</td>" \
             "<td style=\"outline: 1px solid #bbb;{}\">{:.2f}</td>" \
             "</tr>".format(
            name,
            numbercellstyle(mydata[key]["mpjpe"], 0, 100),
            mydata[key]["mpjpe"],
            numbercellstyle(mydata[key]["pmpjpe"], 0, 80),
            mydata[key]["pmpjpe"],
            numbercellstyle(mydata[key]["trajectory"], 0, 100),
            mydata[key]["trajectory"],
            "",
            "",
            "",
            "",
            numbercellstyle(mydata[key]["pck30"], 1, 0),
            mydata[key]["pck30"] * 100,
            numbercellstyle(mydata[key]["pck50"], 1, 0.2),
            mydata[key]["pck50"] * 100,
            numbercellstyle(mydata[key]["pck100"], 1, 0.6),
            mydata[key]["pck100"] * 100,
            numbercellstyle(mydata[key]["pck150"], 1, 0.8),
            mydata[key]["pck150"] * 100,
            numbercellstyle(mydata[key]["pck250"], 1, 0.9),
            mydata[key]["pck250"] * 100,
            numbercellstyle(mydata[key]["p_pck30"], 1, 0),
            mydata[key]["p_pck30"] * 100,
            numbercellstyle(mydata[key]["p_pck50"], 1, 0.2),
            mydata[key]["p_pck50"] * 100,
            numbercellstyle(mydata[key]["p_pck100"], 1, 0.6),
            mydata[key]["p_pck100"] * 100,
            numbercellstyle(mydata[key]["p_pck150"], 1, 0.8),
            mydata[key]["p_pck150"] * 100,
            numbercellstyle(mydata[key]["p_pck250"], 1, 0.9),
            mydata[key]["p_pck250"] * 100,
        )

    '''
    pck_table += "<tr style=\"font-weight:bold\"><td>TOTAL</td><td></td><td></td><td></td><td></td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td>".format(
        tot["PCK30"]/cnt["PCK30"]*100,
        tot["PCK50"]/cnt["PCK50"]*100,
        tot["PCK100"]/cnt["PCK100"]*100,
        tot["PCK250"]/cnt["PCK250"]*100
    )
    pck_table += "<td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>".format(
        tot["PPCK30"]/cnt["PPCK30"]*100,
        tot["PPCK50"]/cnt["PPCK50"]*100,
        tot["PPCK100"]/cnt["PPCK100"]*100,
        tot["PPCK250"]/cnt["PPCK250"]*100
    )
    '''

    pck_table += "</tbody>"
    latex1 = "<textarea readonly>S9, S11&{:.1f}&{:.1f}&{:.1f}&{:.1f}&{:.1f}&{:.1f}&{:.1f}&{:.1f}\\\\</textarea>".format(mydata["TOTALVAL"]["mpjpe"],
        mydata["TOTALVAL"]["pmpjpe"],
        mydata["TOTALVAL"]["pck50"] * 100,
        mydata["TOTALVAL"]["pck100"] * 100,
        mydata["TOTALVAL"]["pck150"] * 100,
        mydata["TOTALVAL"]["p_pck50"] * 100,
        mydata["TOTALVAL"]["p_pck100"] * 100,
        mydata["TOTALVAL"]["p_pck150"] * 100,
    )
    latex2 = "<textarea readonly>Full&{:.1f}&{:.1f}&{:.1f}&{:.1f}&{:.1f}&{:.1f}&{:.1f}&{:.1f}\\\\</textarea>".format(
        mydata["TOTAL"]["mpjpe"],
        mydata["TOTAL"]["pmpjpe"],
        mydata["TOTAL"]["pck50"] * 100,
        mydata["TOTAL"]["pck100"] * 100,
        mydata["TOTAL"]["pck150"] * 100,
        mydata["TOTAL"]["p_pck50"] * 100,
        mydata["TOTAL"]["p_pck100"] * 100,
        mydata["TOTAL"]["p_pck150"] * 100,
        )
    htmlbody += "<h2>Results</h2><table>{}</table>{}{}".format(pck_table,latex1,latex2)

    configTable = "<h2>Full Config</h2>"
    for sec in config.sections():
        configTable += "<h2>"+sec+"</h2><table>"
        for k,v in config[sec].items():
            configTable += "<tr><td>{}</td><td>{}</td><tr>".format(k,v)
        configTable += "</table>"


    with open(os.path.join(config["exec"]["outpath"],'results.html'), 'w') as f:
        f.write("<html><head><div style=\"overflow: auto;\">{}</div></head><body>{}{}</body></html>".format(htmlhead, htmlbody,configTable))
    '''
    
    
    
    ############ ENDE #############
    
    keys = list(a.subject_list)
    keys.sort(key=lambda x: int(x[1:]))
    for subject_ in keys:
        items = list(a.action_list(subject_))
        items.sort(key=lambda x: x[0])
        for action in items:
            gt = np.empty((0))#gt_[subject_][action]
            print("PCK30\t{}\t{}\t{:.2f}\t{}\tframes".format(subject_, action, (ck30[subject_][action].sum()) / len(ck30[subject_][action]) * 100, gt.shape[0]))
    
    
    print("####################################")
    
    keys = list(a.subject_list)
    keys.sort(key=lambda x: int(x[1:]))
    for subject_ in keys:
        items = list(a.action_list(subject_))
        items.sort(key=lambda x: x)
        for action in items:
            gt = np.empty((0))# gt_[subject_][action]
            print("PCK50\t{}\t{}\t{:.2f}\t{}\tframes".format(subject_, action, (ck50[subject_][action].sum()) / len(ck50[subject_][action]) * 100, gt.shape[0]))
    
    print("####################################")
    
    keys = list(a.subject_list)
    keys.sort(key=lambda x: int(x[1:]))
    for subject_ in keys:
        items = list(a.action_list(subject_))
        items.sort(key=lambda x: x)
        for action in items:
            gt = np.empty((0))# gt_[subject_][action]
            print("PCK100\t{}\t{}\t{:.2f}\t{}\tframes".format(subject_, action,
                                                         (ck100[subject_][action].sum()) / len(ck100[subject_][action]) * 100,
                                                         gt.shape[0]))
    
    print("####################################")
    
    keys = list(a.subject_list)
    keys.sort(key=lambda x: int(x[1:]))
    for subject_ in keys:
        items = list(a.action_list(subject_))
        items.sort(key=lambda x: x)
        for action in items:
            gt = np.empty((0))# gt_[subject_][action]
            print("PPCK30\t{}\t{}\t{:.2f}\t{}\tframes".format(subject_, action, (p_ck30[subject_][action].sum()) / len(p_ck30[subject_][action]) * 100, gt.shape[0]))
    
    
    print("####################################")
    
    keys = list(a.subject_list)
    keys.sort(key=lambda x: int(x[1:]))
    for subject_ in keys:
        items = list(a.action_list(subject_))
        items.sort(key=lambda x: x)
        for action in items:
            gt = np.empty((0))# gt_[subject_][action]
            print("PPCK50\t{}\t{}\t{:.2f}\t{}\tframes".format(subject_, action, (p_ck50[subject_][action].sum()) / len(p_ck50[subject_][action]) * 100, gt.shape[0]))
    
    print("####################################")
    
    keys = list(a.subject_list)
    keys.sort(key=lambda x: int(x[1:]))
    for subject_ in keys:
        items = list(a.action_list(subject_))
        items.sort(key=lambda x: x)
        for action in items:
            gt = np.empty((0))# gt_[subject_][action]
            print("PPCK100\t{}\t{}\t{:.2f}\t{}\tframes".format(subject_, action,
                                                          (p_ck100[subject_][action].sum()) / len(p_ck100[subject_][action]) * 100,
                                                          gt.shape[0]))
    print("####################################")
    
    print("PCK Totals")
    
    tot = defaultdict(lambda: 0)
    cnt = defaultdict(lambda: 0)
    
    keys = list(a.subject_list)
    keys.sort(key=lambda x: int(x[1:]))
    for subject_ in keys:
        items = list(a.action_list(subject_))
        items.sort(key=lambda x: x)
        tot["PCK30"] += ck30[subject_][action].sum()
        tot["PCK50"] += ck50[subject_][action].sum()
        tot["PCK100"] += ck100[subject_][action].sum()
        tot["PCK250"] += ck250[subject_][action].sum()
        tot["PPCK30"] += p_ck30[subject_][action].sum()
        tot["PPCK50"] += p_ck50[subject_][action].sum()
        tot["PPCK100"] += p_ck100[subject_][action].sum()
        tot["PPCK250"] += p_ck250[subject_][action].sum()
    
        cnt["PCK30"] += len(ck30[subject_][action])
        cnt["PCK50"] += len(ck50[subject_][action])
        cnt["PCK100"] += len(ck100[subject_][action])
        cnt["PCK250"] += len(ck250[subject_][action])
        cnt["PPCK30"] += len(p_ck30[subject_][action])
        cnt["PPCK50"] += len(p_ck50[subject_][action])
        cnt["PPCK100"] += len(p_ck100[subject_][action])
        cnt["PPCK250"] += len(p_ck250[subject_][action])
    
    
    print("PCK30: {}, PCK50: {}, PCK100: {}, PCK250: {}".format(tot["PCK30"]/cnt["PCK30"]*100,tot["PCK50"]/cnt["PCK50"]*100,tot["PCK100"]/cnt["PCK100"]*100,tot["PCK250"]/cnt["PCK250"]*100))
    print("PPCK30: {}, PPCK50: {}, PPCK100: {}, PCK250: {}".format(tot["PPCK30"]/cnt["PPCK30"]*100,tot["PPCK50"]/cnt["PPCK50"]*100,tot["PPCK100"]/cnt["PPCK100"]*100,tot["PPCK250"]/cnt["PPCK250"]*100))
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #predictions = a["positions_3d"].item()
    
    #person = "S1"
    #action = "Directions 1"
    
    person = "S6"
    action = "Walking"
    
    a = PickleFolder(config, "sequence3D")
    predseq = a.get_sequence(person,action)
    predseq.interpolateCenterFromLeftRight(Feature.hip)
    predseq.interpolateCenterFromLeftRight(Feature.shoulder)
    
    b = PickleFolder(config, "aligned3d")
    predali = b.get_sequence(person,action)
    predali.interpolateCenterFromLeftRight(Feature.hip)
    predali.interpolateCenterFromLeftRight(Feature.shoulder)
    
    gtseq = human36mGT.get_sequence(person,action)
    
    c1 = predseq.cameras[0]
    c2 = predseq.cameras[1]
    
    
    keypoint_iter = list(set(predseq.npdimensions[NpDimension.KEYPOINT_ITER]).intersection(gtseq.npdimensions[NpDimension.KEYPOINT_ITER]))
    keypoint_iter.sort(key=lambda x: (int(x != Keypoint(Position.center, Feature.hip))))
    frame_iter = list(set(predseq.npdimensions[NpDimension.FRAME_ITER]).intersection(gtseq.npdimensions[NpDimension.FRAME_ITER]))
    
    frame_iter.sort()
    
    prediction, f = predseq.get(OrderedDict([(NpDimension.FRAME_ITER,frame_iter),
                                                     (NpDimension.KEYPOINT_ITER,keypoint_iter),
                                                     (NpDimension.KP_DATA, (KpDim.x, KpDim.y, KpDim.z))]))
    prediction_aligned, f = predali.get(OrderedDict([(NpDimension.FRAME_ITER,frame_iter),
                                                     (NpDimension.KEYPOINT_ITER,keypoint_iter),
                                                     (NpDimension.KP_DATA, (KpDim.x, KpDim.y, KpDim.z))]))
    gt, f_ = gtseq.get(OrderedDict([(NpDimension.FRAME_ITER, frame_iter),
                                      (NpDimension.KEYPOINT_ITER, keypoint_iter),
                                      (NpDimension.KP_DATA, (KpDim.x, KpDim.y, KpDim.z))]))
    
    prediction[:,:] -= prediction[:,(0,)]
    prediction_aligned[:,:] -= prediction_aligned[:,(0,)]
    gt[:, :] -= gt[:, (0,)]
    
    #P1 = camera_to_world(predictions[person][action]["P1"][None,:,3],c1.orientation.astype(float),c1.translation)
    #P2 = camera_to_world(predictions[person][action]["P2"][None,:,3],c1.orientation.astype(float),c1.translation)
    
    
    
    
    i=0
    x = prediction_aligned[i, :, 0]
    y = prediction_aligned[i, :, 1]
    z = prediction_aligned[i, :, 2]
    
    x_ = gt[i, :, 0]
    y_ = gt[i, :, 1]
    z_ = gt[i, :, 2]
    
    x__ = prediction[i, :, 0]
    y__ = prediction[i, :, 1]
    z__ = prediction[i, :, 2]
    
    #data, = ax.plot(x,y,z, linestyle="", marker=".", color="blue")
    data2, = ax.plot(x_,y_,z_, linestyle="", marker = ".", color="red")
    data3, = ax.plot(x__,y__,z__, linestyle="", marker = ".", color="yellow")
    txt = list()
    for p in range(prediction_aligned.shape[1]):
        pt = prediction_aligned[i,p]
        txt.append(ax.text(pt[0],pt[1],pt[2],p.__str__()))
    
    for i in range(1,prediction_aligned.shape[0]):
    
        x = prediction_aligned[i, :, 0]
        y = prediction_aligned[i, :, 1]
        z = prediction_aligned[i, :, 2]
    
        x_ = gt[i, :, 0]
        y_ = gt[i, :, 1]
        z_ = gt[i, :, 2]
    
        x__ = prediction[i, :, 0]
        y__ = prediction[i, :, 1]
        z__ = prediction[i, :, 2]
    
        scale = 2 * np.max([1,
                            #np.max(x), np.max(-x), np.max(y), np.max(-y), np.max(z), np.max(-z),
                            np.max(x_), np.max(-x_), np.max(y_), np.max(-y_), np.max(z_), np.max(-z_),
                            np.max(x__), np.max(-x__), np.max(y__), np.max(-y__), np.max(z__), np.max(-z__),
                            ])
        ax.set_xlim3d([-0.5 * scale, 0.5 * scale])
        ax.set_ylim3d([-0.5 * scale, 0.5 * scale])
        ax.set_zlim3d([0, scale])
    
        for i in range(len(x)):
            txt[i].set_position_3d([x[i],y[i],z[i]])
        #data.set_data(x, y)
        #data.set_3d_properties(z)
        data2.set_data(x_, y_)
        data2.set_3d_properties(z_)
        data3.set_data(x__, y__)
        data3.set_3d_properties(z__)
    
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.01)
    
    a.close
    '''
if __name__ == "__main__":

    #main(sys.argv[1])
    files = easygui.fileopenbox(default=os.path.join(os.path.dirname(sys.argv[0]),"*.conf"),filetypes=[["*.conf", "Configuration File"]],multiple=True)
    if files is None:
        exit()
    if isinstance(files,list):
        for f in files:
            main(f)
    else:
        main(files)
