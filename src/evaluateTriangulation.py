from configparser import ConfigParser
import sys

import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib

from external.VideoPose3D.common.camera import normalize_screen_coordinates, camera_to_world
from external.VideoPose3D.common.h36m_dataset import Human36mDataset
from external.VideoPose3D.common.loss import mpjpe, p_mpjpe
from src.h36m_noTears import Human36mNoTears

matplotlib.use("TkAgg")

config = ConfigParser()
if len(sys.argv) != 2:
    print("Wrong number of arguments")
    exit()
config.read(sys.argv[1])

source = "Human3.6m"

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
    subjects = list(keypoints.keys())

    cam1_idx = 0
    cam2_idx = 1

    for s in subjects:
        kp1_[s] = dict()
        kp2_[s] = dict()
        gt_[s] = dict()
        c1[s] = dataset.cameras()[s][cam1_idx]
        c2[s] = dataset.cameras()[s][cam2_idx]

        for k in keypoints[s].keys():
            if cam1_idx in keypoints[s][k] and cam2_idx in keypoints[s][k]:
                kp1_[s][k] = np.array(keypoints[s][k][cam1_idx])
                kp2_[s][k] = np.array(keypoints[s][k][cam2_idx])
                try:
                    gt_[s][k] = np.array(dataset._data[s][k]["positions"])
                    gt_[s][k][:,0] = np.nan
                    gt_[s][k] = gt_[s][k][:,(9,0,0,0,0,14,11,15,12,16,13,1,4,2,5,3,6),:]
                except:
                    print("GT values for {} {} do not exist".format(s,k))
                    gt_[s][k] = None

if source ==  "Human3.6m":

    poseset = "2D_gt"

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

a = np.load("3D-Estimation_Human3.6m_2D_cpn.npz", allow_pickle=True)

#3D-Estimation_Human3.6m_2D_gt
notears = False

predictions = a["positions_3d"].item()

if notears:
    for k in predictions.keys():
        for person, prediction in predictions[k].items():
            gt_[k][person] = gt_[k][person][:prediction["positions_aligned"].shape[0] * 5:5,
            #gt_[k][person] = gt_[k][person][:,
                                      (9, 14, 15, 16, 11, 12, 13, 1, 2, 3, 4, 5, 6), :]
            prediction["positions_aligned"] = prediction["positions_aligned"][:, (0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15), :]

keys = list(predictions.keys())
keys.sort(key=lambda x: int(x[1:]))
for k in keys:
    items = list(predictions[k].items())
    items.sort(key=lambda x: x[0])
    for person, prediction in items:
        gt = gt_[k][person]
        print("P-MPJPE;{};{};{:.2f};mm;{};frames".format(k, person,
                                                         p_mpjpe(
                                                             (prediction["positions_aligned"][:, :, :]) * np.array([1, 1, 1]),
                                                             (gt[:, :, :])) * 1000, gt.shape[0]))

print("####################################")



keys = list(predictions.keys())
keys.sort(key=lambda x: int(x[1:]))
for k in keys:
    items = list(predictions[k].items())
    items.sort(key=lambda x: x[0])
    for person, prediction in items:
        gt = gt_[k][person]
        prediction["positions_aligned"][prediction["positions_aligned"] > 1e20] = np.nan
        prediction["positions_aligned"][prediction["positions_aligned"] <-1e20] = np.nan
        print("MPJPE;{};{};{:.2f};mm;{};frames".format(k, person,
                                                         mpjpe(torch.from_numpy(
                                                             prediction["positions_aligned"][:, :, :]) * np.array([1, 1, 1]),
                                                               (gt[:, :, :])) * 1000, gt.shape[0]))



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

predictions = a["positions_3d"].item()

person = "S1"
action = "Waiting 1"
c1 = dataset.cameras()[person][0]
c2 = dataset.cameras()[person][1]

P1 = camera_to_world(predictions[person][action]["P1"][None,:,3],c1["orientation"].astype(float),c1["translation"])
P2 = camera_to_world(predictions[person][action]["P2"][None,:,3],c1["orientation"].astype(float),c1["translation"])


i=0
x = predictions[person][action]["positions_aligned"][i, :, 0]
y = predictions[person][action]["positions_aligned"][i, :, 1]
z = predictions[person][action]["positions_aligned"][i, :, 2]

x_ = gt_[person][action][i, :, 0]
y_ = gt_[person][action][i, :, 1]
z_ = gt_[person][action][i, :, 2]

x__ = predictions[person][action]["positions"][i, :, 0]
y__ = predictions[person][action]["positions"][i, :, 1]
z__ = predictions[person][action]["positions"][i, :, 2]

data, = ax.plot(x,y,z, linestyle="", marker=".", color="blue")
data2, = ax.plot(x_,y_,z_, linestyle="", marker = ".", color="red")
data3, = ax.plot(x__,y__,z__, linestyle="", marker = ".", color="yellow")
txt = list()
for p in range(predictions[person][action]["positions_aligned"].shape[1]):
    pt = predictions[person][action]["positions_aligned"][i,p]
    txt.append(ax.text(pt[0],pt[1],pt[2],p.__str__()))

for i in range(1,predictions[person][action]["positions_aligned"].shape[0]):

    x = predictions[person][action]["positions_aligned"][i, :, 0]
    y = predictions[person][action]["positions_aligned"][i, :, 1]
    z = predictions[person][action]["positions_aligned"][i, :, 2]

    x_ = gt_[person][action][i, :, 0]
    y_ = gt_[person][action][i, :, 1]
    z_ = gt_[person][action][i, :, 2]

    x__ = predictions[person][action]["positions"][i, :, 0]
    y__ = predictions[person][action]["positions"][i, :, 1]
    z__ = predictions[person][action]["positions"][i, :, 2]

    scale = 2 * np.max([1,
                        np.max(x), np.max(-x), np.max(y), np.max(-y), np.max(z), np.max(-z),
                        np.max(x_), np.max(-x_), np.max(y_), np.max(-y_), np.max(z_), np.max(-z_),
                        np.max(x__), np.max(-x__), np.max(y__), np.max(-y__), np.max(z__), np.max(-z__),])
    ax.set_xlim3d([-0.5 * scale, 0.5 * scale])
    ax.set_ylim3d([-0.5 * scale, 0.5 * scale])
    ax.set_zlim3d([0, scale])

    for i in range(len(x)):
        txt[i].set_position_3d([x[i],y[i],z[i]])
    data.set_data(x, y)
    data.set_3d_properties(z)
    data2.set_data(x_, y_)
    data2.set_3d_properties(z_)
    data3.set_data(x__, y__)
    data3.set_3d_properties(z__)

    plt.show(block=False)
    fig.canvas.draw()
    plt.pause(0.01)

a.close