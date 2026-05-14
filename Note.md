### New Start（2025.10.14）

继续去年的遗志，从今天开始完善项目&论文，且从今日起项目改名为MVCCR-1.0，该版本作为最base的版本。









### Date: 10.21

文件夹名称更新为：ComGCL-v1.3

**问题：**现在对于VGAE部分中生成视图似乎有点不明白，主要是“生成的视图似乎和VGAE部分的训练没有关系”







### Date: 10.19

实现了Embedding的结果进行可视化，如AdaGCL中的那种，代码存放在 `./Visualize/Visualiza.py` 中。

并且对于噪声/稀疏实验结果的折线图也可以用代码来实现，代码存放在 `./Visualize/Draw_double.py` 中。







### Date: 10.01

将训练集中的Embedding保存下来，想用T-SEN来可视化。修改部分代码：

```python
if tstFlag:
    reses, embeds = self.testEpoch()
    if (reses['Recall'] > recallMax):
    recallMax = reses['Recall']
    ndcgMax = reses['NDCG']
    bestEpoch = ep
    torch.save(embeds, r'./Saved/Epoch_'+str(ep)+'.pt')  # 添加的
log(self.makePrint('Test', ep, reses, tstFlag))
print()


ret['Recall'] = epRecall / num
ret['NDCG'] = epNdcg / num
embeds = torch.concat([usrEmbeds, itmEmbeds], dim=0)  # 添加的
return ret, embeds  # 添加的
```







### Date: 09.30

添加多视图对比损失（在主视图BPR之前，生成视图对比损失之后）：

```python
loss_ib = self.model.loss_graphcl(_out1, out1.detach(), ancs, poss) + self.model.loss_graphcl(_out2, out2.detach(), ancs, poss)
loss = loss_ib.mean() * args.ib_reg
ib_loss += float(loss)
loss.backward()

# 尝试在这里引入多视图对比
view0 = self.model.getEmbeds()
# 这里超参的影响很大
loss = (self.model.loss_graphcl(view0, out1.detach(), ancs, poss) + self.model.loss_graphcl(view0, out2.detach(), ancs, poss)).mean() * 0.5
loss.backward()

# BPR  这里计算的是主视图的 BPR loss

```

结果似乎也真的很nice:

```python
-- 0.5
Best epoch :  195  , Recall :  0.268811246996731  , NDCG :  0.19831830833126354
            
-- 0.8
Best epoch :  173  , Recall :  0.2673364255219095  , NDCG :  0.1962449325577384

-- 0.7
Best epoch :  199  , Recall :  0.26780518764389744  , NDCG :  0.1961909155647569

-- 0.6
Best epoch :  197  , Recall :  0.2688597230129489  , NDCG :  0.19743627960802707
                 
-- lastfm topk40            
Best epoch :  177  , Recall :  0.3675819416545224  , NDCG :  0.22995473997645735

-- yelp topk20
Best epoch :  116  , Recall :  0.0940046623742456  , NDCG :  0.04752671435281824
```









### Date: 09.30

重新理解代码意义，并将部分代码进行重构，结果似乎比之前要更加好了：

```python
Best epoch :  138  , Recall :  0.26670711009420695  , NDCG :  0.19464385245971189
```

将项目名称修改为：`ComGCL+MultiViewCL`。

下一步思路：添加多个视图之间的损失，原-view1以及原-view2。那么这样也可以算作是一个创新点了。

<del>另外，在SVD之中生成的两个Embedding实际上并没有进行一个对比损失的计算，只是计算了BPR Loss。</del>

实际上通过：

```python
self.model.loss_graphcl(_out2, out2.detach(), ancs, poss)
```

计算了的！！！



那么下一步工作的重点就在验证多个视图之间对比损失的有效性了。
