#-------------------------------------
# Project: Transductive Propagation Network for Few-shot Learning
# Date: 2019.1.11
# Author: Yanbin Liu
# All Rights Reserved
#-------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
class CNNEncoder(nn.Module):
    """Encoder for feature embedding"""
    def __init__(self, args):
        super(CNNEncoder, self).__init__()
        self.args = args
        h_dim, z_dim = args['h_dim'], args['z_dim']
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

    def forward(self,x):
        """x: bs*3*84*84 """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out 


class RelationNetwork(nn.Module):
    """Graph Construction Module"""
    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,1,kernel_size=3,padding=1),
                        nn.BatchNorm2d(1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))

        self.fc3 = nn.Linear(2*2, 8)
        self.fc4 = nn.Linear(8, 1)

        self.m0 = nn.MaxPool2d(2)            # max-pool without padding 
        self.m1 = nn.MaxPool2d(2, padding=1) # max-pool with padding

    def forward(self, x, rn):
        
        x = x.view(-1,64,5,5) # (100, 64, 5, 5)
        
        out = self.layer1(x) # （100, 64, 3, 3）
        out = self.layer2(out) # (100, 1, 2, 2)
        # flatten
        out = out.view(out.size(0),-1) # (100 ,4)
        out = F.relu(self.fc3(out)) # (100 ,8)
        out = self.fc4(out) # (100, 1)

        out = out.view(out.size(0),-1) # (100, 1)

        return out


class Prototypical(nn.Module):
    """Main Module for prototypical networlks"""
    def __init__(self, args):
        super(Prototypical, self).__init__()
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.h_dim, self.z_dim = args['h_dim'], args['z_dim']

        self.args = args
        self.encoder = CNNEncoder(args)

    def forward(self, inputs):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        [support, s_labels, query, q_labels] = inputs
        num_classes = s_labels.shape[1]
        num_support = int(s_labels.shape[0] / num_classes)
        num_queries = int(query.shape[0] / num_classes)

        inp   = torch.cat((support,query), 0)
        emb   = self.encoder(inp) # 80x64x5x5
        emb_s, emb_q = torch.split(emb, [num_classes*num_support, num_classes*num_queries], 0)
        emb_s = emb_s.view(num_classes, num_support, 1600).mean(1)
        emb_q = emb_q.view(-1, 1600)
        emb_s = torch.unsqueeze(emb_s,0)     # 1xNxD
        emb_q = torch.unsqueeze(emb_q,1)     # Nx1xD
        dist  = ((emb_q-emb_s)**2).mean(2)   # NxNxD -> NxN

        ce = nn.CrossEntropyLoss().cuda()
        loss = ce(-dist, torch.argmax(q_labels,1))
        ## acc
        pred = torch.argmax(-dist,1)
        gt   = torch.argmax(q_labels,1)
        correct = (pred==gt).sum()
        total   = num_queries*num_classes
        acc = 1.0 * correct.float() / float(total)

        return loss, acc


class LabelPropagation(nn.Module):
    """Label Propagation"""
    def __init__(self, args):
        super(LabelPropagation, self).__init__()
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.h_dim, self.z_dim = args['h_dim'], args['z_dim'] # h_dim: 64, z_dim: h_dim

        self.args = args
        self.encoder = CNNEncoder(args)
        self.relation = RelationNetwork()

        if   args['rn'] == 300:   # learned sigma, fixed alpha
            self.alpha = torch.tensor([args['alpha']], requires_grad=False)
        elif args['rn'] == 30:    # learned sigma, learned alpha
            self.alpha = nn.Parameter(torch.tensor([args['alpha']]), requires_grad=True)

    def forward(self, inputs):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        # init
        eps = torch.tensor(torch.finfo(torch.float32).eps).cuda()

        [support, s_labels, query, q_labels] = inputs
        num_classes = s_labels.shape[1] # 5
        num_support = int(s_labels.shape[0] / num_classes) # 5
        num_queries = int(query.shape[0] / num_classes) # 15

        # Step1: Embedding
        inp     = torch.cat((support,query), 0) # (100, 3, 84, 84) 将suport和query set concat在一块
        emb_all = self.encoder(inp).reshape(-1,1600) # (100, 1600) 合并在一起提取特征
        N, d    = emb_all.shape[0], emb_all.shape[1]

        # Step2: Graph Construction
        ## sigmma
        if self.args['rn'] in [30,300]:
            self.sigma   = self.relation(emb_all, self.args['rn']) # (100, 1)
            
            ## W
            emb_all = emb_all / (self.sigma+eps) # N*d -> (100, 1600)
            emb1    = torch.unsqueeze(emb_all,1) # N*1*d
            emb2    = torch.unsqueeze(emb_all,0) # 1*N*d
            W       = ((emb1-emb2)**2).mean(2)   # N*N*d -> N*N，实现wij = (fi - fj)**2
            W       = torch.exp(-W/2)

        ## keep top-k values
        if self.args['k']>0:
            topk, indices = torch.topk(W, self.args['k']) # topk: (100, 20), indices: (100, 20)
            mask = torch.zeros_like(W).cuda()
            mask = mask.scatter(1, indices, 1) # (100, 100)
            mask = ((mask+torch.t(mask))>0).type(torch.float32) #torch.t() 期望 input 为<= 2-D张量并转置尺寸0和1。   # union, kNN graph
            #mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
            W    = W*mask # 构建无向图，上面的mask是为了保证把wij和wji都保留下来

        ## normalize
        D       = W.sum(0) # (100, )
        D_sqrt_inv = torch.sqrt(1.0/(D+eps)) # (100, )
        D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,N) # (100, 100)
        D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(N,1) # (100, 100)
        S       = D1*W*D2

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        ys = s_labels # (25, 5)
        yu = torch.zeros(num_classes*num_queries, num_classes).cuda() # (75, 5)
        #yu = (torch.ones(num_classes*num_queries, num_classes)/num_classes).cuda()
        y  = torch.cat((ys,yu),0) # (100, 5)用supoort set的label去预测query的label
        F  = torch.matmul(torch.inverse(torch.eye(N).cuda() - self.alpha*S+eps), y) # (100, 5)
        Fq = F[num_classes*num_support:, :] # query predictions，loss计算support和query set一起算，acc计算只计算query
        
        # Step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss().cuda()
        ## both support and query loss
        gt = torch.argmax(torch.cat((s_labels, q_labels), 0), 1)
        loss = ce(F, gt)
        ## acc
        predq = torch.argmax(Fq,1)
        gtq   = torch.argmax(q_labels,1)
        correct = (predq==gtq).sum()
        total   = num_queries * num_classes
        acc = 1.0 * correct.float() / float(total)

        return loss, acc
