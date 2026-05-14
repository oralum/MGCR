from torch import nn
import torch.nn.functional as F
import torch
from Params import args
from copy import deepcopy
import numpy as np
import math
import scipy.sparse as sp
from Utils.Utils import contrastLoss, calcRegLoss, pairPredict, sparse_dropout

import torch_sparse

init = nn.init.xavier_uniform_

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

	def forward_gcn(self, adj):  # 将经过每一层GCN后的Embedding存储，最后将所有层的Embedding进行相加，将userEmbedding和itemEmbedding分别返回。
		iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

		embedsLst = [iniEmbeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])  # 当前的GCN使用的是上一层GCN产生的Embedding
			embedsLst.append(embeds)
		mainEmbeds = sum(embedsLst)

		return mainEmbeds[:args.user], mainEmbeds[args.user:]

	def forward_graphcl(self, adj):  # 与上一个相同，但是将所有的Embedding全部全返回
		iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

		embedsLst = [iniEmbeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		mainEmbeds = sum(embedsLst)

		return mainEmbeds

	def forward_graphcl_(self, generator):  # 本质和上面的GCN差不多，只不过这里多了一个generator来产生ADJ，并使用这个ADJ来进行GCN操作。
		iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

		embedsLst = [iniEmbeds]		
		count = 0
		for gcn in self.gcnLayers:
			with torch.no_grad():
				adj = generator.generate(x=embedsLst[-1], layer=count)
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
			count += 1
		mainEmbeds = sum(embedsLst)

		return mainEmbeds

	def loss_graphcl(self, x1, x2, users, items):
		T = args.temp
		user_embeddings1, item_embeddings1 = torch.split(x1, [args.user, args.item], dim=0)  # (1892, 128)  (17632, 128)
		user_embeddings2, item_embeddings2 = torch.split(x2, [args.user, args.item], dim=0)  #

		user_embeddings1 = F.normalize(user_embeddings1, dim=1)
		item_embeddings1 = F.normalize(item_embeddings1, dim=1)
		user_embeddings2 = F.normalize(user_embeddings2, dim=1)
		item_embeddings2 = F.normalize(item_embeddings2, dim=1)

		user_embs1 = F.embedding(users, user_embeddings1)  # (4096, 128)
		item_embs1 = F.embedding(items, item_embeddings1)  # (4096, 128)
		user_embs2 = F.embedding(users, user_embeddings2)  # embedding() 表示 embedding 函数会根据 users 张量中的每个索引值，从 user_embeddings1 嵌入矩阵中检索对应的行，并返回一个新的张量，其中包含了这些索引对应的嵌入向量。
		item_embs2 = F.embedding(items, item_embeddings2)

		all_embs1 = torch.cat([user_embs1, item_embs1], dim=0)  # (8192, 128)
		all_embs2 = torch.cat([user_embs2, item_embs2], dim=0)  # (8192, 128)

		all_embs1_abs = all_embs1.norm(dim=1)  # (8192, )用于计算张量 all_embs1 在指定维度上的范数，默认是  L2 范数（也称为欧几里得范数），它等同于向量各元素平方和的平方根。（dim=1）
		all_embs2_abs = all_embs2.norm(dim=1)  # (8192, )

		"""
		torch.einsum('ik,jk->ij', all_embs1, all_embs2)：
			这个操作使用爱因斯坦求和约定计算 all_embs1 和 all_embs2 之间的点积。结果是一个矩阵，其中每个元素 (i, j) 是 all_embs1 中第 i 个向量
			和 all_embs2 中第 j 个向量的点积。这个矩阵表示了 all_embs1 中每个向量与 all_embs2 中每个向量的相似度。
		torch.einsum('i,j->ij', all_embs1_abs, all_embs2_abs)：
			这个操作计算 all_embs1 和 all_embs2 中向量的范数的外积。结果是一个矩阵，其中每个元素 (i, j) 是 all_embs1 中第 i 个向量和 all_embs2 中
			第 j 个向量的范数的乘积。这个矩阵用于归一化点积，以得到余弦相似度。
		这两部分的除法得到余弦相似度矩阵，其中每个元素 (i, j) 是 all_embs1 中第 i 个向量和 all_embs2 中第 j 个向量的余弦相似度。
		"""
		sim_matrix = torch.einsum('ik,jk->ij', all_embs1, all_embs2) / torch.einsum('i,j->ij', all_embs1_abs, all_embs2_abs)
		sim_matrix = torch.exp(sim_matrix / T)
		pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]
		loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
		loss = - torch.log(loss)

		return loss

	def getEmbeds(self):
		self.unfreeze(self.gcnLayers)
		return torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

	def unfreeze(self, layer):
		for child in layer.children():
			for param in child.parameters():
				param.requires_grad = True

	def getGCN(self):
		return self.gcnLayers

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds, flag=True):
		if (flag):
			return torch.spmm(adj, embeds)
		else:
			return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)

	# 对 GCN 进行改造
	# def forward(self, adj, embeds, flag=True):
	# 	x = F.dropout(embeds, p=0.20, training=True)
	# 	x = F.normalize(x, p=2, dim=0)
	# 	if (flag):
	# 		z = (1 - 0.005 * 100 + 0.005 * 0.003) * x
	# 		s = 0.5 * torch.spmm(adj, x)
	# 		t = (0.005 * 0.003) * x @ (x.t() @ x)
	# 		return z + s - t
	# 	else:
	# 		z = (1 - 0.005 * 100 + 0.005 * 0.003) * x
	# 		s = (0.005 * 100) * torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], x)
	# 		t = (0.005 * 0.003) * x @ (x.t() @ x)
	# 		return z + s - t

class vgae_encoder(Model):
	def __init__(self):
		super(vgae_encoder, self).__init__()
		hidden = args.latdim
		self.encoder_mean = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
		self.encoder_std = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.Softplus())

	def forward(self, adj):  # 这个方法是用于变分图自编码器（VGAE）中的编码器部分的前向传播，它模拟计算了数据的均值和标准差，然后通过添加噪声来生成新的数据点。
		x = self.forward_graphcl(adj)

		x_mean = self.encoder_mean(x)  # 模拟嵌入的均值
		x_std = self.encoder_std(x)  # 模拟嵌入的方差（由于方差必须得为正数，因此最后使用了一个softplus()）
		gaussian_noise = torch.randn(x_mean.shape).cuda()
		x = gaussian_noise * x_std + x_mean
		return x, x_mean, x_std

class vgae_decoder(nn.Module):
	def __init__(self, hidden=args.latdim):
		super(vgae_decoder, self).__init__()
		self.decoder = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
		self.sigmoid = nn.Sigmoid()
		self.bceloss = nn.BCELoss(reduction='none')  # 二元交叉熵损失函数，用于衡量模型预测的概率分布与真实标签之间的差异。

	def forward(self, x, x_mean, x_std, users, items, neg_items, encoder):
		x_user, x_item = torch.split(x, [args.user, args.item], dim=0)

		edge_pos_pred = self.sigmoid(self.decoder(x_user[users] * x_item[items]))
		edge_neg_pred = self.sigmoid(self.decoder(x_user[users] * x_item[neg_items]))

		loss_edge_pos = self.bceloss(edge_pos_pred, torch.ones(edge_pos_pred.shape).cuda())  # 计算正样本的二元交叉熵损失
		loss_edge_neg = self.bceloss(edge_neg_pred, torch.zeros(edge_neg_pred.shape).cuda())  # 计算负样本的二元交叉熵损失
		loss_rec = loss_edge_pos + loss_edge_neg

		kl_divergence = - 0.5 * (1 + 2 * torch.log(x_std) - x_mean**2 - x_std**2).sum(dim=1)

		ancEmbeds = x_user[users]
		posEmbeds = x_item[items]
		negEmbeds = x_item[neg_items]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
		regLoss = calcRegLoss(encoder) * args.reg
		
		beta = 0.1
		loss = (loss_rec + beta * kl_divergence.mean() + bprLoss + regLoss).mean()
		
		return loss

class vgae(nn.Module):
	def __init__(self, encoder, decoder):
		super(vgae, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, data, users, items, neg_items):
		x, x_mean, x_std = self.encoder(data)
		loss = self.decoder(x, x_mean, x_std, users, items, neg_items, self.encoder)
		return loss

	def generate(self, data, edge_index, adj):
		x, _, _ = self.encoder(data)  # x 为生成的新Embedding

		edge_pred = self.decoder.sigmoid(self.decoder.decoder(x[edge_index[0]] * x[edge_index[1]]))  # decoder()就是一个输入为latdim，输出为1的MLP，用于进行预测。

		vals = adj._values()  # (149490, )
		idxs = adj._indices()  # (2, 149490)
		edgeNum = vals.size()
		edge_pred = edge_pred[:, 0]
		mask = ((edge_pred + 0.5).floor()).type(torch.bool)  # (149490, ) 对边预测结果进行阈值处理（在这里是 0.5），并转换为布尔值，用于过滤预测为正的边。
		
		newVals = vals[mask]  # (6252) 使用 mask 过滤邻接矩阵中的值，只保留预测为正的边的权重。

		newVals = newVals / (newVals.shape[0] / edgeNum[0])  # 对过滤后的边权重进行归一化处理
		newIdxs = idxs[:, mask]  # 选择 idxs 中所有列的元素，但仅限于 mask 中对应位置为 True 的行
		
		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)  # 创建一个新的稀疏邻接矩阵，并返回。这个新矩阵只包含预测为正的边。


class SVDNet(Model):
	def __init__(self, adj, svd_q):
		super(SVDNet, self).__init__()
		self.adj = adj.cuda()
		self.q = svd_q
		self.split_adj = self.splitAdj(self.adj)
		self.E_u_list = [None] * (args.gnn_layer+1) # gnn_layer+1个数组
		self.E_i_list = [None] * (args.gnn_layer+1)
		self.Z_u_list = [None] * (args.gnn_layer+1)
		self.Z_i_list = [None] * (args.gnn_layer+1)
		self.G_u_list = [None] * (args.gnn_layer+1)
		self.G_i_list = [None] * (args.gnn_layer+1)

	def splitAdj(self, adj):  # 拆分邻接矩阵
		adj = adj.to_dense()
		m1, _ = torch.split(adj, [args.user, args.item], dim=0)
		_, x1 = torch.split(m1, [args.user, args.item], dim=1)
		# print(x1.shape)
		return x1.to_sparse(sparse_dim=2)

	def generate(self):
		self.embeds = self.forward_graphcl(self.adj)
		self.embeds = self.embeds.to('cuda:0')
		self.svd_u, s, self.svd_v = torch.svd_lowrank(self.split_adj, self.q)
		self.u_mul_s = self.svd_u @ (torch.diag(s))  # 将s对角矩阵乘到svd_u上，即将奇异值重新组合到左奇异向量上。
		self.v_mul_s = self.svd_v @ (torch.diag(s))  # 将s对角矩阵乘到svd_v上，即将奇异值重新组合到右奇异向量上。

		self.uEmb, self.iEmb = torch.split(self.embeds, [args.user, args.item], dim=0)  # (1892, 32)   (17632, 32)
		self.E_u_list[0] = self.uEmb
		self.E_i_list[0] = self.iEmb
		self.G_u_list[0] = self.uEmb
		self.G_i_list[0] = self.iEmb

		for layer in range(1, args.gnn_layer+1):
			self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.split_adj, args.dropout), self.E_i_list[layer - 1]))  # Light GCN传播
			self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.split_adj, args.dropout).transpose(0, 1), self.E_u_list[layer - 1]))

			vt_ei = self.svd_v.T @ self.E_i_list[layer - 1]  # 左乘原右奇异矩阵
			self.G_u_list[layer] = (self.u_mul_s @ vt_ei)  # 左乘左
			ut_eu = self.svd_u.T @ self.E_u_list[layer - 1]
			self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

			# aggregate
			self.E_u_list[layer] = self.Z_u_list[layer]
			self.E_i_list[layer] = self.Z_i_list[layer]

		self.G_u = sum(self.G_u_list)
		self.G_i = sum(self.G_i_list)
		self.G_emb = torch.concat([self.G_u, self.G_i], axis=0)

		# aggregate across layers
		self.E_u = sum(self.E_u_list)
		self.E_i = sum(self.E_i_list)
		self.E_emb = torch.concat([self.E_u, self.E_i], axis=0)

		return self.G_emb, self.E_emb

		# 由于这里使用的是方阵图，若将图按行和列进行分解则难以进行SVD传播
		# embList = [self.linear(emb)]
		# for layer in range(args.gnn_layer):
		# 	emb = torch.spmm(self.adj, embList[-1])
		# 	vt = self.svd_v.T @ emb
		# 	us = self.u_mul_s @ vt
		# 	ut = self.svd_u.T @ us
		# 	vs = self.v_mul_s @ ut
		# 	embList.append(vs)
		# 	layer += 1
		# self.svd_emb = sum(embList)
		# return F.relu(self.svd_emb, inplace=True)

	def forward(self, users, items, neg_items):  # 这里算Loss的时候会出问题
		emb = self.G_emb.detach()
		x_user, x_item = torch.split(emb, [args.user, args.item], dim=0)
		ancEmbeds = x_user[users]
		posEmbeds = x_item[items]
		negEmbeds = x_item[neg_items]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - scoreDiff.sigmoid().log().sum() / args.batch
		regLoss = calcRegLoss(self) * args.reg

		return bprLoss