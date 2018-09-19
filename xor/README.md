# xor的一些说明

用来解决异或问题的是一个多层感知机。
- 代码里写的是两层神经网络，中间层的神经元个数为2,最后一层的神经元个数为1
- 激活函数采用的是sigmoid函数
- 学习率设置为0.05
- loss采用的是l2 loss， 即 MSE
- 最后将计算图用 `tf.summary.FileWriter` 保存，如果查看的话，在终端输入
```
tensorboard --logdir='./graph'
```
会显示端口号，然后直接在浏览器输入localhost:port便可以看到计算图。
