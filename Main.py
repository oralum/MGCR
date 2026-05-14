import torch

import Ali_Uni
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, vgae_encoder, vgae_decoder, vgae, SVDNet
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import calcRegLoss, pairPredict
import os
from copy import deepcopy
import scipy.sparse as sp
import random
import datetime


class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')

		recallMax = 0
		ndcgMax = 0
		bestEpoch = 0

		stloc = 0
		log('Model Initialized')

		save_path = f'./Saved/{str(args.data)}/top_{str(args.topk)}/bestEmbeds_{str(args.latdim)}.pt'
		save_dir = os.path.dirname(save_path)

		for ep in range(stloc, args.epoch):
			temperature = max(0.05, args.init_temperature * pow(args.temperature_decay, ep))
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch(temperature)
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				reses, embeds = self.testEpoch()
				if (reses['Recall'] > recallMax):
					recallMax = reses['Recall']
					ndcgMax = reses['NDCG']
					bestEpoch = ep
					if not os.path.exists(save_dir):
						os.makedirs(save_dir)
					# 现在可以安全地保存文件
					torch.save(embeds, save_path)
				log(self.makePrint('Test', ep, reses, tstFlag))
			print()
		print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax)


		# 获取当前时间并格式化为"年月日时分"的格式
		current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
		# 打开文件以写入内容，如果文件不存在将会创建它
		with open('./Saved/result.txt', 'a') as file:
			# 将格式化后的时间和要保存的文本写入文件
			file.write("======== " + str(args.data) + "   " + str(args.latdim) + "   " + str(args.gnn_layer) + "   " + str(args.topk) + "   " + str(args.temp) + "   " + " ========" + '\n')
			file.write('Current time : ' + current_time + '\n')
			file.write('Best epoch: ' + str(bestEpoch) + ', Recall: ' + str(recallMax) + ', NDCG: ' + str(ndcgMax) + '\n\n')

	def prepareModel(self):
		self.model = Model().cuda()

		encoder = vgae_encoder().cuda()
		decoder = vgae_decoder().cuda()
		self.generator_1 = vgae(encoder, decoder).cuda()
		self.generator_2 = SVDNet(deepcopy(self.handler.torchBiAdj), 11).cuda()  # 这里的 5 就是 LightGCL中的q

		self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
		self.opt_gen_1 = torch.optim.Adam(self.generator_1.parameters(), lr=args.lr, weight_decay=0)
		self.opt_gen_2 = torch.optim.Adam(self.generator_2.parameters(), lr=args.lr, weight_decay=0)  # 12.10  如果只是进行一个svd是不是就不需要优化器了？从而将SVDNet中的线性层也可以删除了？

	def trainEpoch(self, temperature):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		generate_loss_1, generate_loss_2, bpr_loss, im_loss, ib_loss, reg_loss = 0, 0, 0, 0, 0, 0
		steps = trnLoader.dataset.__len__() // args.batch

		for i, tem in enumerate(trnLoader):
			data = deepcopy(self.handler.torchBiAdj).cuda()

			data1 = self.generator_generate(self.generator_1)  # data1是经 VGAE 的 Encoder 部分生成的邻接矩阵

			self.opt.zero_grad()
			self.opt_gen_1.zero_grad()
			self.opt_gen_2.zero_grad()

			ancs, poss, negs = tem
			ancs = ancs.long().cuda()  # user节点
			poss = poss.long().cuda()  # 正 item 节点
			negs = negs.long().cuda()  # 负 item 节点

			out1 = self.model.forward_graphcl(data1)  # 通过生成的邻接矩阵来获得Embedding
			# 09.30：这里generator_2得输出是不是得重新考虑一下？（再参考下LightGCL论文）
			out2, _ = self.generator_2.generate()

			# 两个生成视图之间的InfoNCE损失（View1 和 View2）
			loss = self.model.loss_graphcl(out1, out2, ancs, poss).mean() * args.ssl_reg
			im_loss += float(loss)
			loss.backward()

			# info bottleneck = InfoNCE + InfoNCE -- 两个视图的对比损失（即View1-Main + View2-Main）
			# 09.30：那么这个这个说法是错误的啊啊啊啊啊！既然以前理解错了，那么或许真的可以试试生成试图与原试图计算一下对比损失试试？
			_out1 = self.model.forward_graphcl(data1)
			_out2, _ = self.generator_2.generate()

			# is_same = torch.equal(out1, _out1)  # 09.30：out1 和 _out1 不相等，这又是何意呢？

			# 09.30：若 out1 和 _out1 不相等，那么这里的 loss 岂不是在计算同一视图生成器生成的不同试图的对比损失？
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
			usrEmbeds, itmEmbeds = self.model.forward_gcn(data)
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]
			scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
			bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
			regLoss = calcRegLoss(self.model) * args.reg

			# 在这里引入AU_loss
			# 09.30：要计算对比学习中Embedding的对齐性和一致性，那么得使用out1和out2进行对比才行吧？亦或是out1和out2分别与原始Embedding相比？
			# ali_loss = alignment(out1[ancs], out2[ancs]) + alignment(out1[poss], out2[poss])
			# uni_loss = uniformity(ancEmbeds, posEmbeds)
			# direct_loss = ali_loss

			loss = bprLoss + regLoss
			bpr_loss += float(bprLoss)
			reg_loss += float(regLoss)
			loss.backward()

			loss_1 = self.generator_1(deepcopy(self.handler.torchBiAdj).cuda(), ancs, poss, negs)  # 生成器内部的损失
			loss_2 = self.generator_2(ancs, poss, negs)

			loss = loss_1 + loss_2
			generate_loss_1 += float(loss_1)
			generate_loss_2 += float(loss_2)
			loss.backward()

			self.opt.step()
			self.opt_gen_1.step()
			self.opt_gen_2.step()

			log('Step %d/%d: gen 1 : %.3f ; gen 2 : %.3f ; bpr : %.3f ; im : %.3f ; ib : %.3f ; reg : %.3f' % (
				i, 
				steps,
				generate_loss_1,
				generate_loss_2,
				bpr_loss,
				im_loss,
				ib_loss,
				reg_loss
				), save=False, oneline=True)

		ret = dict()
		ret['Gen_1 Loss'] = generate_loss_1 / steps
		ret['Gen_2 Loss'] = generate_loss_2 / steps
		ret['BPR Loss'] = bpr_loss / steps
		ret['IM Loss'] = im_loss / steps
		ret['IB Loss'] = ib_loss / steps
		ret['Reg Loss'] = reg_loss / steps

		return ret

	def testEpoch(self):
		tstLoader = self.handler.tstLoader
		epRecall, epNdcg = [0] * 2
		i = 0
		num = tstLoader.dataset.__len__()  # 获取数据集中元素的总数
		steps = num // args.tstBat  # 数据总量 / batch大小 得到 steps
		for usr, trnMask in tstLoader:  # tstLoader返回的俩是什么？-----> 去看 TstData 类中的 __getitem__ 方法！
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()
			usrEmbeds, itmEmbeds = self.model.forward_gcn(self.handler.torchBiAdj)
			allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8  # 1 - trnMask 将 trnMask 中的非零元素转换为0，零元素转换为1，从而得到一个掩码，其中用户已经交互过的物品位置为0，未交互过的位置为1，目的是在计算最终的预测分数时忽略用户已经交互过的物品。
			_, topLocs = torch.topk(allPreds, args.topk)		  															   # - trnMask * 1e8 将用户已经交互过的物品的预测分数设置为一个非常低的值（接近负无穷）。这样做是为了在后续的排序或选择 top-k 推荐时，这些物品不会出现在推荐列表中。
			recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)					   # 这行代码的目的是计算用户对物品的预测评分，并使用掩码来调整分数，确保用户已经交互过的物品在推荐时不会被考虑。这是一种常见的处理方法，用于在推荐系统中生成不包含已知用户偏好的推荐列表。
			epRecall += recall
			epNdcg += ndcg
			log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		embeds = torch.concat([usrEmbeds, itmEmbeds], dim=0)
		return ret, embeds

	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))  # 这个将索引+2的操作，应该是出于计算逻辑上的考虑，毕竟0+1=1，log(0)和log(1)都没有意义，所以不如直接+2
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			allRecall += recall
			allNdcg += ndcg
		return allRecall, allNdcg

	def generator_generate(self, generator):
		edge_index = []
		edge_index.append([])
		edge_index.append([])
		adj = deepcopy(self.handler.torchBiAdj)
		idxs = adj._indices()

		with torch.no_grad():
			view = generator.generate(self.handler.torchBiAdj, idxs, adj)

		return view


if __name__ == '__main__':
	with torch.cuda.device(args.gpu):
		logger.saveDefault = True
		
		log('Start')
		handler = DataHandler()
		handler.LoadData()
		log('Load Data')

		coach = Coach(handler)
		coach.run()
