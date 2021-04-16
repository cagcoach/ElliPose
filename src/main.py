# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys, os
import numpy as np

from external.VideoPose3D.common.visualization import render_animation

sys.path.append(os.path.abspath('external/VideoPose3D'))
from external.VideoPose3D.common.camera import normalize_screen_coordinates
from external.VideoPose3D.common.h36m_dataset import Human36mDataset
import cv2


from configparser import ConfigParser

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



    kp1 = keypoints["S1"]["Directions"][0]
    kp2 = keypoints["S1"]["Directions"][1]

    m = min(kp1.shape[0], kp2.shape[0]);

    kp1 = kp1[:m,:,:]
    kp2 = kp2[:m,:,:]

    kp1 = kp1.reshape(-1, 2)
    kp2 = kp2.reshape(-1, 2)

    kp1[12][1] = 11237. # Artificial outlier

    c1 = dataset.cameras()["S1"][0]
    c2 = dataset.cameras()["S1"][1]

    cameraMatrix1 = np.array( [[c1["focal_length"][0],                      0, c1["center"][0]],
                               [                    0,  c1["focal_length"][1], c1["center"][1]],
                               [                    0,                      0,               1]])

    cameraMatrix2 = np.array([[c2["focal_length"][0], 0, c2["center"][0]],
                              [0, c2["focal_length"][1], c2["center"][1]],
                              [0, 0, 1]])

    E=cv2.findEssentialMat(kp1, kp2, cameraMatrix1=cameraMatrix1, cameraMatrix2=cameraMatrix2, distCoeffs1=None,
                         distCoeffs2=None, method=cv2.RANSAC, prob=0.5, threshold=0.0001)

    #fundmat = cv2.findFundamentalMat(kp1,kp2,cv2.FM_RANSAC, ransacReprojThreshold=0.0001, confidence=0.999)

    retval, R, t, mask, triangulatedPoints = cv2.recoverPose(E[0], kp1, kp2, cameraMatrix=cameraMatrix1, distanceThresh=10);

    real_camera0_orientation = dataset.cameras()["S1"][0]["orientation"]
    real_camera0_translation = dataset.cameras()["S1"][0]["translation"]

    real_camera1_orientation = dataset.cameras()["S1"][1]["orientation"]
    real_camera1_translation = dataset.cameras()["S1"][1]["translation"]

    prediction = triangulatedPoints.transpose()[:, :3].reshape(-1, 17, 3)
    input_keypoints = keypoints["S1"]["Directions"][0]
    prediction -= np.mean(prediction,(0,1));
    prediction *= 1/np.max(prediction)
    prediction[:,:,1] *= -1
    keypoints_metadata = keypoints2D['metadata'].item()
    keypoints_metadata["layout_name"] = ""
    render_animation(input_keypoints[::4,:,:], keypoints_metadata, {'Reconstruction': prediction[::4,:,(2,0,1)]}, dataset.skeleton(),dataset.fps()/4, 3000,00,"out0_"+poseset+".mp4",viewport=(cam['res_w'], cam['res_h']))
    render_animation(input_keypoints[::4,:,:], keypoints_metadata, {'Reconstruction': prediction[::4,:,(0,2,1)]}, dataset.skeleton(),dataset.fps()/4, 3000,00,"out1"+poseset+".mp4",viewport=(cam['res_w'], cam['res_h']))
    render_animation(input_keypoints[::4,:,:], keypoints_metadata, {'Reconstruction': prediction[::4,:,(0,1,2)]}, dataset.skeleton(),dataset.fps()/4, 3000,00,"out2"+poseset+".mp4",viewport=(cam['res_w'], cam['res_h']))
    render_animation(input_keypoints[::4,:,:], keypoints_metadata, {'Reconstruction': prediction[::4,:,(1,2,0)]}, dataset.skeleton(),dataset.fps()/4, 3000,00,"out3"+poseset+".mp4",viewport=(cam['res_w'], cam['res_h']))



    kp3d = dataset._data["S1"]["Directions"]["positions"].reshape(-1,3)

    print(retval)


    #hd = Human36mDataset(path = config.get("Human3.6m","Location"))
    #print(hd._data)



    # Use a breakpoint in the code line below to debug your script.
      # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    main(sys.argv)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
