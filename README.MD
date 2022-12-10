# RoboCup

- Branch of yolov7.
- A repo of RoboCup.（中国机器人大赛先进视觉赛项）
-
- rosws为项目工程文件，为identify赛项与measure赛项服务.
- identify赛项分为yolov7与yolov5两个版本
- yolov5版本更稳定
- yolov7版本识别准确率更高但是在jetson nano上更消耗算力
- measure赛项依赖一种神奇的方式运行
-
- TrainTools为神经网络训练的一些脚本化的工具，为数据集的准备，整理，增强提供便利。

- mv_driver包：相机驱动模块，用于将深度相机SR300的RGB图像以及深度图像从USB端口接受后实时通过
rostopic发布出去，发布节点：/cloud, /raw_img, /raw_img_depth
mv_driver包的实现难点：将SR300中获取到的点云数据，深度图和RGB图像压缩并实时发送到计算平台。
需要保证低时延，低损耗的特性。
