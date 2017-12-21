## notMNIST 手写字母识别

##### 该项目主要使用 tensorflow 对 not mnist 数据集进行手写字母识别

> 目录结构
- load.py: 加载数据的基类; 同时也是下载数据的基类；对外提供 nextBatch 接口
- base.py: 所有分类器都会继承该基类；里面封装了各种常用函数
- logistic.py: 逻辑回归
- bp_simply.py: bp 神经网络
- bp_with_tensorboard.py: bp 神经网络，运行过程的数据以及效果会记录到 tensorboard 中
- cnn_simply.py: cnn 卷积神经网络；同样使用了 tensorboard

<br>

> 准确率
>- cnn_simply.py:
>>- training set accuracy: 0.978598%
>>- validation set accuracy: 0.940516%
>>- test set accuracy: 0.939964%

<br>
