from external.VideoPose3D.common.mocap_dataset import MocapDataset


class Estimated3dDataset(MocapDataset):
    def __init__(self, estimatorInput, estimatorOutput,skeleton):
        super().__init__(fps=50,skeleton=skeleton)
        assert len(estimatorInput) == len(estimatorOutput)


        self._data = dict()
        self._cameras = dict()

        for i in range(len(estimatorInput)):
            kp1 = estimatorInput[i][0]
            kp2 = estimatorInput[i][1]
            gt = estimatorInput[i][2]
            c1 = estimatorInput[i][3]
            c2 = estimatorInput[i][4]
            k = estimatorInput[i][5]
            s = estimatorInput[i][6]

            prediction = estimatorOutput[i][0]
            newprediction = estimatorOutput[i][1]
            gt_ = estimatorOutput[i][2]
            k_ = estimatorOutput[i][3]
            s_ = estimatorOutput[i][4]
            P = estimatorOutput[i][5]
            P_ = estimatorOutput[i][6]
            #assert((gt_ == gt).any() and (k_ == k).any() and (s_ == s).any())

            if not s in self._data:
                self._data[s] = {}
            if not s in self._cameras:
                self._cameras[s] = [c1,c2]
            self._data[s][k] = {
                "positions": newprediction
            }