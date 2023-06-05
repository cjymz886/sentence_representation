# sentence_representation
some training tricks for  sentence representation

**本项目初衷是为了想验证cosent损失函数+SSCL对比损失，是否会更利于有监督句子表征学习，同时也把其他训练trick是进行融合测试，使用以下了训练策略：**
- [苏神提出的cosent损失函数](https://github.com/bojone/CoSENT)
- [缓解句子过度平滑的对比损失-SSCL](https://zhuanlan.zhihu.com/p/630520876)
- 对抗训练-PGD
- 防止过拟合-EMA
- 防止过拟合-R_Dropout
- 优化器-Lookahead

除了SSCL是我根据论文思路重构了下，其他都是现有方法的组合，有一些tricks也是打比赛常用的方法。

# About data
项目只在LCQMC数据集上进行测试，其训练集、验证集、测试集分别为：238766、8802、12500。

# About training
- 为了跟之前相关工作对比，训练的常见参数基本保持一致，如seq_len=64, epochs=5，预训练模型为roberta，句子向量使用cls方式等；
- 由于涉及不同的组合问题，本次并没有逐一验证每个trick是否有对结果有提升，只是验证了以下几种情况：（1）只采用cosent损失函数，其可以看着是样本对之间的对比学习；（2）cosent损失函数+SSCL对比损失，其加上了样本之间的对比损失，并从模型倒数第二层采用作为负样本；（3）cosent+SSCL+PGD+EMA+R_Dropout,增加三种正则化训练策略；三种对比都是在Lookahead优化器上。
- 训练：python run.py train
- 测试：python run.py test

# About experiment
评价指标都是采用斯皮尔曼系数评价，在测试集上实验结果如下：
| model | LCQMC|
| ------| ------|
|Roberta+CoSENT(苏神)|79.31|
|Roberta+CoSENT(shawroad)|79.38|
|Roberta+CoSENT(our)|78.93|
|Roberta+CoSENT+SSL(our)|79.11|
|Roberta+CoSENT+mix(our)|79.77|

其中mix为PGD+EMA+R_Dropout。可以看出：
- 在原始Roberta+CoSENT模型上，我跑出的结果都没有超过苏神和shawroad实验的结果；
- 在使用SSL对比损失后，实验结果略有提升；
- 在使用所有tricks后，斯皮尔曼系数达到79.77，感觉提升效果还是可以的；


# Conclusion
由于本项目实验做的并不是很充分，只是局部验证了SSCL对比损失是能带来一定效果的，同时也验证了多个训练tricks融合也是能带来可观的提升；若有兴趣，可以尝试不同组合，或微调下其中的超参数，在其他数据集上进行尝试；

# Reference
1.[bojone/CoSENT](https://github.com/bojone/CoSENT)<br>
2.[shawroad/CoSENT_Pytorch](https://github.com/shawroad/CoSENT_Pytorch)<br>
3.[一种缓解无监督句子表征的过度平滑的方法](https://zhuanlan.zhihu.com/p/630520876)<br>
