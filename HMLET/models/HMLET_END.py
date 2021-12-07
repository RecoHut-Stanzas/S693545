import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np

from .gating.gating_network import Gating_Net

class BasicModel(nn.Module):    
	def __init__(self):
		super(BasicModel, self).__init__()

	def getUsersRating(self, users):
		raise NotImplementedError

class HMLET_End(nn.Module):
	def __init__(self, 
					config:dict, 
					dataset:BasicDataset):
		super(HMLET_End, self).__init__()
		self.config = config
		self.dataset : dataloader.BasicDataset = dataset
		self.__init_model()

	def __init_model(self):
		self.num_users = self.dataset.n_users
		self.num_items = self.dataset.m_items
		self.embedding_dim = self.config['embedding_dim']
   
		self.n_layers = 4
		self.dropout = self.config['dropout']
		self.keep_prob = self.config['keep_prob']
		self.A_split = self.config['a_split']

		# Embedding
		self.embedding_user = torch.nn.Embedding(
			num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
		self.embedding_item = torch.nn.Embedding(
			num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
		
		# Normal distribution initilizer
		nn.init.normal_(self.embedding_user.weight, std=0.1)
		nn.init.normal_(self.embedding_item.weight, std=0.1)      
		
		# Activation function
		selected_activation_function = self.config['activation_function']
		
		if selected_activation_function == 'relu':
			self.r = nn.ReLU()
			self.activation_function = self.r
		if selected_activation_function == 'leaky-relu':
			self.leaky = nn.LeakyReLU(0.1)
			self.activation_function = self.leaky
		elif selected_activation_function == 'elu':
			self.elu = nn.ELU()
			self.activation_function = self.elu
		print('activation_function:',self.activation_function)
		
		self.g_train = self.dataset.getSparseGraph()

		# Gating Net with Gumbel-Softmax
		self.gating_network_list = []
		for i in range(2):
			self.gating_network_list.append(Gating_Net(embedding_dim=self.embedding_dim, mlp_dims=self.config['gating_mlp_dims']).to(world.device))

	def __choosing_one(self, features, gumbel_out):
		feature = torch.sum(torch.mul(features, gumbel_out), dim=1)  # batch x embedding_dim (or batch x embedding_dim x layer_num)
		return feature

	def __dropout_x(self, x, keep_prob):
		size = x.size()
		index = x.indices().t()
		values = x.values()
		random_index = torch.rand(len(values)) + keep_prob
		random_index = random_index.int().bool()
		index = index[random_index]
		values = values[random_index]/keep_prob
		g = torch.sparse.FloatTensor(index.t(), values, size)
		return g

	def __dropout(self, keep_prob):
		if self.A_split:   
			graph = []
			for g in self.Graph:
				graph.append(self.__dropout_x(g, keep_prob))
		else:
			graph = self.__dropout_x(self.Graph, keep_prob)
		return graph

	def computer(self, gum_temp, hard):     
		
		self.Graph = self.g_train   
		if self.dropout:
			if self.training:
				g_droped = self.__dropout(self.keep_prob)
			else:
				g_droped = self.Graph        
		else:
			g_droped = self.Graph
    
    
		# Init users & items embeddings  
		users_emb = self.embedding_user.weight
		items_emb = self.embedding_item.weight
      
      
		## Layer 0
		all_emb_0 = torch.cat([users_emb, items_emb])
		
		# Residual embeddings
		embs = [all_emb_0]
		
   
		## Layer 1
		all_emb_lin_1 = torch.sparse.mm(g_droped, all_emb_0)
		
		# Residual embeddings	
		embs.append(all_emb_lin_1)
		
   
		## layer 2
		all_emb_lin_2 = torch.sparse.mm(g_droped, all_emb_lin_1)
		
		# Residual embeddings
		embs.append(all_emb_lin_2)
		
   
		## layer 3
		all_emb_lin_3 = torch.sparse.mm(g_droped, all_emb_lin_2)
		all_emb_non_1 = self.activation_function(torch.sparse.mm(g_droped, all_emb_0))
		
		# Gating
		stack_embedding_1 = torch.stack([all_emb_lin_3, all_emb_non_1],dim=1)
		concat_embeddings_1 = torch.cat((all_emb_lin_3, all_emb_non_1),-1)

		gumbel_out_1, lin_count_3, non_count_3 = self.gating_network_list[0](concat_embeddings_1, gum_temp, hard, self.config['division_noise'])
		embedding_1 = self.__choosing_one(stack_embedding_1, gumbel_out_1)

		# Residual embeddings
		embs.append(embedding_1)
	
  	
		# layer 4
		all_emb_lin_4 = torch.sparse.mm(g_droped, embedding_1)
		all_emb_non_2 = self.activation_function(torch.sparse.mm(g_droped, embedding_1))
		
		# Gating
		stack_embedding_2 = torch.stack([all_emb_lin_4, all_emb_non_2],dim=1)
		concat_embeddings_2 = torch.cat((all_emb_lin_4, all_emb_non_2),-1)

		gumbel_out_2, lin_count_4, non_count_4 = self.gating_network_list[1](concat_embeddings_2, gum_temp, hard, self.config['division_noise'])
		embedding_2 = self.__choosing_one(stack_embedding_2, gumbel_out_2)

		# Residual embeddings  		
		embs.append(embedding_2)


		## Stack & mean residual embeddings
		embs = torch.stack(embs, dim=1)
		light_out = torch.mean(embs, dim=1)
   
		users, items = torch.split(light_out, [self.num_users, self.num_items])
		
		return users, items, [lin_count_3, non_count_3, lin_count_4, non_count_4], embs

	def getUsersRating(self, users, gum_temp, hard):
		all_users, all_items, gating_dist, embs = self.computer(gum_temp, hard)
		
		users_emb = all_users[users.long()]
		items_emb = all_items

		rating = self.activation_function(torch.matmul(users_emb, items_emb.t()))

		return rating, gating_dist, embs

	def getEmbedding(self, users, pos_items, neg_items, gum_temp, hard):
		all_users, all_items, gating_dist, embs = self.computer(gum_temp, hard)
		
		users_emb = all_users[users]
		pos_emb = all_items[pos_items]
		neg_emb = all_items[neg_items]

		users_emb_ego = self.embedding_user(users)
		pos_emb_ego = self.embedding_item(pos_items)
		neg_emb_ego = self.embedding_item(neg_items)

		return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, gating_dist, embs

	def bpr_loss(self, users, pos, neg, gum_temp, hard):
		(users_emb, pos_emb, neg_emb, 
		userEmb0,  posEmb0, negEmb0, gating_dist, embs) = self.getEmbedding(users.long(), pos.long(), neg.long(), gum_temp, hard)
		
		reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
							posEmb0.norm(2).pow(2)  +
							negEmb0.norm(2).pow(2))/float(len(users))
		
		pos_scores = torch.mul(users_emb, pos_emb)
		pos_scores = torch.sum(pos_scores, dim=1)
		neg_scores = torch.mul(users_emb, neg_emb)
		neg_scores = torch.sum(neg_scores, dim=1)
		
		loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
		
		return loss, reg_loss, gating_dist, embs
		
	def forward(self, users, items, gum_temp, hard):
		# compute embedding
		all_users, all_items, gating_dist, embs = self.computer(gum_temp, hard)

		users_emb = all_users[users]
		items_emb = all_items[items]

		inner_pro = torch.mul(users_emb, items_emb)
		gamma     = torch.sum(inner_pro, dim=1)

		return gamma, gating_dist, embs