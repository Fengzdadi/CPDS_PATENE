# Patent

## mine

### datasets

For introduction, please refer to the article *A pose-based*

+ RVI-38
  + 38个不同的3至5个月大的婴儿
  + 手持Sony DSC-RX100
  + 分辨率为1920 × 1080 @ 25 FPS
  + 每个视频的时长在40秒到5分钟
  + 平均时长为3分钟36秒

+ MINI-RGBD
  + 每个视频时长40秒
  + 12个从上往下拍摄的、处于仰卧位置的婴儿视频

## A Pose-based feature fusion and classification

### Methodology

#### Proposed Framework Overview

+ 传统的机器学习算法
+ 原始视频作为输入，计算关节位置
+ 关节位置纠正 见下
+ 基于GMA（General Movements Assessment）的特征生成
+ 分类框架  见下
+ 特征融合 见下

#### Pose Estimation and Data Pre-Processing

+ **姿态估计**
  + OpenPose 提取关节位置
  + （x和y）坐标和一个关联的预测置信度得分表示
+ **数据预处理** 具体公式见文章
  + 自动数据校正
    + 利用预测置信度计算置信度阈值 得分计算 -5%，逐帧移除置信度低于置信度低于阈值的关节位置
    + 异常的关节，通过相邻两帧进行插值（AKIMA），移动平均滤波器 减少序列抖动，经验表明 5 帧能得到较好的结果
  + 旋转规范化坐标
    + 使用关节 9 作为根，修正所有关节，根固定在0,0的位置
    + 关节2，9 为脊柱，对齐脊柱位置，旋转 $$/theta$$  

#### Feature Extraction

+ **角位移**
  + 角位移代表了视频中每个身体部位在指定时间间隔内的角定向变化。这种基于直方图的特征捕捉了预定义规则偏移间隔之间的角位移分布，从而可以表示身体部位运动的平滑度。帮助识别短时的痉挛、突然和零星的运动
  + 父关节指向子关节的2D向量
  + 使得该特征只关注定向变化的幅度
+ **相对关节方向**
  + 使用基于直方图的特征来表示关节的相对方向分布
  + 通过计算两个关节之间的相对方向来实现这一目标
  + 提取了单个关节的直方图后，合并以形成每个肢体的直方图表示
  + 直观地表示了身体的同步性
  + 如果一个直方图只有少数几个箱子具有高值，意味着关节是同时朝着同一方向移动

+ **相对关节角位移**
  + 评估身体部位之间的关系，评估整体身体协调性、肌张力障碍和共济失调运动。
  + 算出的大多数角位移具有小的值 实施非均匀箱子大小以增加HORJAD2D特征的区分能力
  + 经验性地发现 当 n*=8 时，可以获得最佳结果。
+ **快速傅立叶变换的关节位移（FFT-JD）**
  + 从运动中提取的每个频率分量的幅值 评估运动的可变性
  + 模拟运动的复杂性、流动性和多样性
  + 突显任何重复、运动障碍、震颤或肌阵挛的特点
+ **快速傅立叶变换的关节方向（FFT-JO）**
  + 婴儿姿势相关的刚度、方向变化和运动范围的信息
  + 以捕捉运动的重复性和周期性
  + 关节方向信号转换为频域，可以评估运动的刚度和流畅度
  + 频率分量，可以识别出重复、僵硬或异常的运动模式
+ **关节方向直方图 (HOJO2D) 和关节位移直方图 (HOJD2D)**
  + 关于各个关节运动和关节之间协调的信息
  + 保留和展现婴儿每个肢体的运动特征

#### **Classification**

+ 特征标准化
  + Z-score 允许系统保留原始数据集的形状属性
+ 分类框架
  + 逻辑回归（LR）、支持向量机（SVM）、决策树（Tree）、线性判别分析（LDA）、分类模型的集合（Ens），以及 k-最近邻（k-Nearest Neighbour）算法
+ 性能评估

#### FeatureFusion

+ 特征融合
  + 将选定的特征融合在一起进行进一步分析
+ 基于姿态的特征
+ 基于速度的特征
+ 融合分类

## A Spatio-temporal Attention-based Model for Infant Movement Assessment from Videos

### Proposed method

#### pipeline overview

+ STAM

#### pose extraction

+ openpose
+ preprocessing
  + 数据填充 缺失数据线性插值填充
  + 离群值移除 够你懂中位数过滤器
  + 信号平滑 滚动平均过滤器（窗口0.5s）
  + 姿态归一化
    + 旋转上身绕肩部
    + 旋转下身绕臀部
    + 躯干长度归一化

#### Motion features computation

+ 运动特征计算 速度 加速度 关节行进距离 
+ 平滑处理 滚动平均滤波器
+ 构建姿态序列
+ 运动特征向量
  + 2D坐标
  + x，y轴上的速度
  + x，y轴上的加速度
  + 第j个关节在帧t上的行进距离

#### STAM：A Spatio-temporal Attention-based Model

+ 概述

  ​	数据准备

  ​	模型构建 不安定运动和不规则运动发生的启发

  ​	给定姿态序列分割成 K个固定长度

  ​	空间注意图卷积网络（sag）

  ​	时间注意机制

+ SAG
  + ST-GEN扩展 在这个模型顶部加入一个空间注意力层扩展ST-GCN，为关节生成注意力权重
  + 关节表征
  + 聚合函数
  + 空间注意力权重 每个关节计算空间注意力权重 双曲正切激活函数 softmax函数
  + 可学习参数
+ Temporal attention-based aggregation
  + 时间注意力基础聚合 从剪辑的表征中使用加权平均作为聚合函数 权重又网络生成
  + 计算时间注意力权重
  + 可学习参数
  + 预测 sigmoid作为激活函数
+ Prediction 
  + sigmoid
+ loss  function 
  + cross-entropy

## Identification of Abnormal Movements in Infants: A Deep Neural Network for Body Part-Based Prediction of Cerebral Palsy

### Abstract

+ GMA(General Movements Assessment) 有希望，但是手动很耗时

+ 可视化框架

### METHODOLOGY-OVERVIEW AND DATA PRE-PROCESSING

+ 姿态划分为不同的身体部分
+ 每个身体部分分成不同分支

#### DATA PRE-PROCESSING

+ POSE ESTIMATION FROM VIDEO
  + 官方的OpenPose实现（https://github.com/CMU-Perceptual-Computing-Lab/openpose）从视频中提取关节的2D位置。具体来说，每个视频都被转换为一系列图像，并从每个图像中提取一个骨架姿态。对于每个姿态，检测到18个关键点，包括身体关节位置和面部标志。图2显示了一个示例。每个关键点包含图像中关节位置的x和y坐标。在这项工作中，使用了14个关节，包括头、颈、左右肩、左右肘、左右腕、左右髋、左右膝和左右踝。
+ AUTOMATIC DATA CORRECTION
  + 数据自动校正 
  + $\ ti=\left(\frac1n\sum_{j=1}^nci,j\right)\times90\%(1) $
  + 90% 作为阈值
  +  不符合的使用Akima插值 样条插值
+ DATA NORMALIZATION
  + 缩放因子以及方向

#### METHODOLOGY-THE PROPOSED FRAMEWORK

+ PART-BASED MOVEMENTMODELLING USING CNN AND LSTM
  + CNN 空间
    + 1D CNN 因为数据量不多，所以更高效
    + 5个小块进行空间建模
    + 批量归一化（减少协方差偏移）
  + LSTM 时间
    + 2块
    + 紧接着 dropout 层（提高繁华能力） + 一个批量归一化
+ BODY-PARTABNORMALITY DETECTION
  + 使用全局最大池化来突显嵌入向量中编码的异常
  + 添加了两个全连接层来编码异常信息
  + 分类层
  + 每个流的输出将是一个贡献分数 指示身体部位对分类决策的贡献有多大
+ CLASSIFICATION
  + 一个丢弃层随机丢弃50%的神经元
  + 
