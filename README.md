# ElliPose

## Dependencies

https://github.com/cagcoach/VideoPose3D

## Running the ElliPose Algorithm

### Interactive

You can run the ElliPose algorithm by running
```
python main.py
```
and then opening a config file using the file explorer window.

### Non Interactive

All settings are managed using a config file.
```
python main.py ../config/default.conf
```

You can also use multiple config files to e.g. seperate configurations from paths
```
python main.py ../config/location.conf ../config/default.conf 
```

by adding ```--continue``` you can continue an interupted session.

## Evaluate Results

After running the ElliPose Algorithm, you will find a config file in the output directory. To analyse the results you need to use this config file by either using the interactive file browser

```
python evaluateTriangulation.py
```
or by directly providing a config file

```
python evaluateTriangulation.py path/to/my/result/config.conf
```

# Publication

Christian Grund, Julian Tanke, and Juergen Gall. "ElliPose: Stereoscopic 3D Human Pose Estimation by Fitting Ellipsoids." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2023.

https://openaccess.thecvf.com/content/WACV2023/html/Grund_ElliPose_Stereoscopic_3D_Human_Pose_Estimation_by_Fitting_Ellipsoids_WACV_2023_paper.html