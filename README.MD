# 面向自动驾驶的目标检测与实现
![image](https://user-images.githubusercontent.com/99338120/230750911-ed89961c-6889-44b4-80a3-be7db6676193.png)


- Branch of yolov7.
- 
-
- rosws为项目工程文件，为identify赛项与measure赛项服务.
- identify赛项分为yolov7与yolov5两个版本
- yolov5版本更稳定
- yolov7版本识别准确率更高但是在jetson nano上更消耗算力
- measure赛项依赖一种神奇的方式运行
-
- TrainTools为神经网络训练的一些脚本化的工具，为数据集的准备，整理，增强提供便利。

- mv_driver包：相机驱动模块，用于将深度相机SR300的RGB图像以及深度图像从USB端口接受后实时通过
rostopic发布出去
- 订阅节点：
- 发布节点：/cloud, /raw_img, /raw_img_depth
- mv_driver包的实现难点：将SR300中获取到的点云数据，深度图和RGB图像压缩并实时发送到计算平台。
需要保证低时延，低损耗的特性。
- 
- plane_extract包：检测平面，通过接收的mv_driver包的点云数据来进行聚类，在点云数据中提取出桌
面的四个角，并通过一系列转置计算以及条件约束推算出具体的桌面在图像中的具体位置，方便后期处理.   
订阅节点：/cloud，/isIdentify    
发布节点：/calibrateResult   
plane_extract包实现的难点：需首先对相机的内参进行标定以方便进行点云数据的坐标系与世界坐标系的转
置。平面检测与提取的过程需要通过多种条件的约束来保证提取的平面可用且稳定性高，SR300提供的点云数据
精度较低且点云范围较小，在对点云数据的提取与提取桌面的条件约束上需要大量的调试。   
 
- scheduler包：对整个程序的具体运行阶段进行管理，统筹安排需要进行与停止的进程。
- 订阅节点：/isIdentify，/calibrateResult，/rcnn_results
- 发布节点：通过对全局广播的动态参数统筹各进程的调度。
- scheduler包实现的难点：通过发布一个/step的节点来统筹所有进程的协调稳定性较低，改使用对全局广播
的动态参数来保证多线程高并发的可靠性
- 
- ui包：qt编写的前端程序，实现了对程序的控制与RGB识别结果和点云的识别结果的可视化。
- 
- video_pub包：用于调试，通过本机的视频文件来模拟SR300输出的RGB图像。
- 发布节点：/raw_img, /raw_img_depth
- video_pub包实现的难点：模拟SR300输出的RGB图像并作为节点发布
- 
- identify包：载入训练完成后并转换的神经网络模型并进行识别，将识别出的平面检测结果与神经网络模型
检测出的物体识别结果比对，过滤贴图与非桌面的物体将识别的结果实时展示在前端上并转储在txt文件中。
- 订阅节点：/raw_img, /isIdentify
- 发布节点：/rcnn_results, /nn_beat
- identify包实现的难点：神经网络模型的训练与简化，物体识别的具体流程
- 
- yolo_identify_v5包：大致功能与identify包类似，但调用的是yolov5的神经网络模型，在pytorch
训练出pt文件后转化为wts文件，然后通过tensorrtx工具转化成engine文件，通过TensorRT对物体的识
别检测加速。
 
