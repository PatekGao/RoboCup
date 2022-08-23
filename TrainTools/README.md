# RoboCup
#### 南京航空航天大学长空御风战队
#### 先进视觉赛项训练用源码

## 部署
> 注意：不推荐直接`pip install -r requirements.txt`安装依赖，此操作可能导致安装出错或者没有正确使用cuda

#### 建议安装步骤
警告： 以下步骤中CUDA、TRT、pytorch的版本和安装方式可能过时，请谨慎采纳
1. 安装CUDA、CUDNN，CUDA版本推荐使用102或者111，其他版本可能导致兼容性问题
2. 安装pytorch，推荐采用1.9版本，若无法支持，次选1.7版本，安装参考[pytorch官网](https://pytorch.org/get-started)
3. 安装detectron2，在windows下，请参考官方文档从源码安装，对于linux用户，推荐采用wheel文件安装，具体参考[detectron2文档](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only)
4. 安装其他依赖项`pip install matplotlib jupyter opencv-python pycocotools`

> 注意：请使用正确的pip进行安装，在部分系统中需要采用pip3以安装至python3环境，请勿直接复制粘贴指令

## 使用
从队伍OSS下载`annotations`和`image`目录下所有压缩包备用，如需使用已训练的模型请额外下载`model.zip`  
下载完成后，在本目录执行`jupyter notebook`启动jupyter，或采用pycharm直接打开工程，进入`notebooks`文件夹，跟随`dataset.ipynb`笔记本的指示部署数据集  
使用`train.ipynb`笔记本进行训练或推理，使用预训练模型请将`model.zip`解压至`notebooks/output`文件夹  

## 工具库指南
* baiduImage.ipynb - 百度图片爬虫
* dataset.ipynb - 数据集解压、合并、去重、转格式工具库，切换识别/测量工作空间
* enhanceIdentify.py - 对识别赛项数据集执行90、180、270度旋转增强
* enhanceMeasure.py - 对测量赛项数据集执行缩放增强
* measureDemo.ipynb - 测量赛项原理演示demo
* shot.py - 数据集拍摄工具
* train.py - MaskRCNN训练笔记本
