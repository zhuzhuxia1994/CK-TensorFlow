# CK-TensorFlow
use TensorFlow and CNN train and test the CK+ dataset

使用双流卷积神经网络对CK+的表情库进行分类，识别结果可以达到99%。CK的表情数据库在网上可以下载，双流网络针对的是RGB图片和光流图片，光流图片需要自己制作，分为X方向光流和Y方向光流

model.py中保存着双流卷积神经网络的模型，这是自己搭建的一个简单的双流卷积神经网络，如果有需要可以进行修改。

train.py是训练，test.py是测试

数据库地址：链接：https://pan.baidu.com/s/1boBvOBX 密码：sem9
