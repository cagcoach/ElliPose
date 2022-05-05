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

styles = \
'''
.subtotal{ background-color:#ddd; }
.subtotal td{ outline: 1px solid #ddd; }
.total{ background-color:#bbb; }
.total td{ outline: 1px solid #bbb; }
.actioncol, .subjectcol{font-weight:bold;}
thead{ position: sticky; top: 0;}
thead tr{background-color:#ddd;}
'''

for colorID in range(256):
    bgcolor = applyColorMap(np.array(colorID).astype(np.uint8), COLORMAP)

    textcolor = "fff" if (bgcolor[0, 0, 2] * 0.2126) + (bgcolor[0, 0, 1] * 0.7152) + (
                bgcolor[0, 0, 0] * 0.0722) < 127 else "000"

    hexbgcolor = bgcolor[0, 0, 0] + 0x100 * bgcolor[0, 0, 1] + 0x10000 * bgcolor[0, 0, 2]
    key = ".b" + str(colorID)
    styles+= key + "{" + "background-color:#{:06x};color:#{};text-align:right;".format( hexbgcolor, textcolor) + "}\n"


def numbercellstyle(val, min_, max_):
    changedval = (val - min_)/(max_-min_) * 127
    colorID = np.array(128 + min(changedval, 127)).astype(np.uint8)
    key = "b" + str(colorID)
    #return "background-color:#{:06x};color:#{};text-align:right".format(hexbgcolor,textcolor)
    return key


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

refA = 0

def generateHtmlDataRow(datadict: dict, subject:str, action:str, htmlclass: str = "") -> str:

    htmlstr = "<tr class=\""+htmlclass+"\">" if (htmlclass != "") else "<tr>"

    # subject
    htmlstr += "<td class=\"subjectcol\">{}</td>".format(subject)
    #action
    htmlstr += "<td class=\"actioncol\">{}</td>".format(action)
    #mpjpe
    htmlstr += "<td class=\"{}\">{:.2f} mm</td>".format(
                     numbercellstyle(datadict["mpjpe"], 0, 100),
                     datadict["mpjpe"],
                 )
    #pmpjpe
    htmlstr += "<td class=\"{}\">{:.2f} mm</td>".format(
        numbercellstyle(datadict["pmpjpe"], 0, 80),
        datadict["pmpjpe"]
    )
    #trajectory
    htmlstr += "<td class=\"{}\">{:.2f} mm</td>".format(
        numbercellstyle(datadict["trajectory"], 0, 100),
        datadict["trajectory"]
    )
    #bestA
    bA = datadict["bestA"] if "bestA" in datadict else np.nan
    htmlstr += "<td class=\"{}\">{:.4f}</td>".format(
        numbercellstyle(np.log10(bA), np.log10(max(0.0001,refA)), 0),
        bA
    )
    #frames
    frms = datadict["frames"] if "frames" in datadict else ""
    htmlstr += "<td>{}</td>".format(
        frms
    )

    htmlstr += "<td class=\"{}\">{:.2f}</td>".format(numbercellstyle(datadict["pck30"], 1, 0),
        datadict["pck30"] * 100,)
    htmlstr += "<td class=\"{}\">{:.2f}</td>".format(numbercellstyle(datadict["pck50"], 1, 0.2),
        datadict["pck50"] * 100,)
    htmlstr += "<td class=\"{}\">{:.2f}</td>".format(numbercellstyle(datadict["pck100"], 1, 0.6),
        datadict["pck100"] * 100,)
    htmlstr += "<td class=\"{}\">{:.2f}</td>".format(numbercellstyle(datadict["pck150"], 1, 0.8),
        datadict["pck150"] * 100,)
    htmlstr += "<td class=\"{}\">{:.2f}</td>".format(numbercellstyle(datadict["pck250"], 1, 0.9),
        datadict["pck250"] * 100,)
    htmlstr += "<td class=\"{}\">{:.2f}</td>".format(numbercellstyle(datadict["p_pck30"], 1, 0),
        datadict["p_pck30"] * 100,)
    htmlstr += "<td class=\"{}\">{:.2f}</td>".format(numbercellstyle(datadict["p_pck50"], 1, 0.2),
        datadict["p_pck50"] * 100,)
    htmlstr += "<td class=\"{}\">{:.2f}</td>".format(numbercellstyle(datadict["p_pck100"], 1, 0.6),
        datadict["p_pck100"] * 100,)
    htmlstr += "<td class=\"{}\">{:.2f}</td>".format(numbercellstyle(datadict["p_pck150"], 1, 0.8),
        datadict["p_pck150"] * 100,)
    htmlstr += "<td class=\"{}\">{:.2f}</td>".format(numbercellstyle(datadict["p_pck250"], 1, 0.9),
        datadict["p_pck250"] * 100,)
    htmlstr += "</tr>\n"

    return htmlstr

def main(conf):
    matplotlib.use("TkAgg")

    config = ConfigParser()

    config.read(conf)

    if not os.path.isdir(config.get("exec","outpath")):
        op = config.get("exec","outpath")
        cp = conf
        strindex = cp.index(op)
        config["exec"]["outpath"] = cp[:(strindex+len(op))]

    human36mGT = Human36mGT(config)

    source = config["exec"]["source"]
    refA = config.getfloat("Ellipse","breakingcondition")
    #dataset = Human36mDataset(config.get(source, "3D"))

    a = PickleFolder(config, "aligned3d")
    htmlhead = "<title>{}</title>".format(config["exec"]["outpath"])
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

    mydata = defaultdict(lambda:defaultdict(lambda:dict()))

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

        mydata[subject_] = defaultdict(lambda:dict())

        items = a.action_list(subject_)
        items.sort(key=lambda x: x)


        def process(action : str):
            gtaction = action.split("_")[0]

            outdict = dict()
            predseq = a.get_sequence(subject_,action)
            bestA = a.get_bestA_for_Sequence(subject_,action)
            predseq.interpolateCenterFromLeftRight(Feature.hip)
            predseq.interpolateCenterFromLeftRight(Feature.shoulder)
            gtseq = human36mGT.get_sequence(subject_,gtaction)
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
    mydata["TOTAL"]["actions"] = defaultdict(lambda:list())

    mydata["TOTALVAL"]["trajectory"] = np.mean([mydata[s]["TOTAL"]["trajectory"] for s in set(["S9", "S11"]).intersection(keys)])
    mydata["TOTALVAL"]["mpjpe"] = np.mean([mydata[s]["TOTAL"]["mpjpe"] for s in set(["S9","S11"]).intersection(keys)])
    mydata["TOTALVAL"]["pmpjpe"] = np.mean([mydata[s]["TOTAL"]["pmpjpe"] for s in set(["S9","S11"]).intersection(keys)])
    mydata["TOTALVAL"]["actions"] = defaultdict(lambda:list())

    tot = defaultdict(lambda: 0)
    cnt = defaultdict(lambda: 0)
    keys = list(a.subject_list)
    keys.sort(key=lambda x: int(x[1:]))
    pck_table = "<thead>" \
                "<tr>" \
                "<td>subject</td>" \
                "<td>action</td>"\
                "<td>MPJPE</td><td>PMPJPE</td><td>Trajectory</td><td>Best A</td><td>frames</td>"\
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
            gtaction = action.split("_")[0]
            mydata["TOTAL"]["actions"][gtaction].append((subject_, action))
            if subject_ in ["S9","S11"]:
                mydata["TOTALVAL"]["actions"][gtaction].append((subject_, action))

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

            pck_table += generateHtmlDataRow(mydata[subject_][action],subject_,action)


        for k in ["pck30", "pck50", "pck100", "pck150", "pck250", "p_pck30", "p_pck50", "p_pck100", "p_pck150", "p_pck250"]:
            mydata[subject_]["TOTAL"][k] = np.mean([mydata[subject_][a][k] for a in items])

    for k in ["pck30","pck50","pck100","pck150","pck250","p_pck30","p_pck50","p_pck100","p_pck150","p_pck250"]:
        mydata["TOTAL"][k] = np.mean([mydata[s]["TOTAL"][k] for s in keys])
        mydata["TOTALVAL"][k] = np.mean([mydata[s]["TOTAL"][k] for s in set(["S9", "S11"]).intersection(keys)])



    for subject_ in keys:
        pck_table += generateHtmlDataRow(mydata[subject_]["TOTAL"], subject_, "TOTAL","subtotal")

    for gtactions in mydata["TOTAL"]["actions"].keys():
        ga = mydata["TOTAL"]["actions"][gtactions]
        pck_table += generateHtmlDataRow({
            key: np.mean([mydata[s][a][key] for s,a in ga]) for key in ["mpjpe","pmpjpe","trajectory","pck30","pck50","pck100","pck150","pck250","p_pck30","p_pck50","p_pck100","p_pck150","p_pck250"]
        }, "ALL", gtactions, "subtotal")

    pck_table += generateHtmlDataRow(mydata["TOTALVAL"], "S9,S11", "TOTAL", "total")
    pck_table += generateHtmlDataRow(mydata["TOTAL"], "ALL", "TOTAL", "total")


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


    with open(os.path.join(os.path.dirname(conf),'results.html'), 'w') as f:
        f.write("<html><head>{}<style>{}</style></head><body>{}{}</body></html>".format(htmlhead,styles, htmlbody,configTable))

if __name__ == "__main__":

    #main(sys.argv[1])
    if len(sys.argv) <= 1:
        files = easygui.fileopenbox(default=os.path.join(os.path.dirname(sys.argv[0]),"*.conf"),filetypes=[["*.conf", "Configuration File"]],multiple=True)
        if files is None:
            exit()
        if isinstance(files,list):
            for f in files:
                main(f)
        else:
            main(files)
    else:
        for file in sys.argv[1:]:
            if os.path.isdir(file):
                files = easygui.fileopenbox(default=os.path.join(file, "*.conf"),
                                    filetypes=[["*.conf", "Configuration File"]], multiple=True)
                if files is None:
                    exit()
                if isinstance(files, list):
                    for f in files:
                        main(f)
                else:
                    main(files)
            else:
                main(file)
