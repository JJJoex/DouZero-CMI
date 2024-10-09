from math import gamma
from sklearn.neighbors import NearestNeighbors
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np
import torch
import random


class ClassifierModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, tau):
        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, num_classes)
        self.tau = tau
    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.h3(x)
        x = F.softmax(x, dim=-1)
        hardT = nn.Hardtanh(self.tau, 1-self.tau)
        x = hardT(x)
        return x
    

def sample_batch(data, batch_size, sample_mode='joint'):
    # get batch data

    if sample_mode == 'joint':
        index = np.random.choice(len(data), batch_size, replace=False)
        batch = np.concatenate([data[i]['value'] for i in index])
    elif sample_mode == 'prod_knn':
        index = np.random.choice(len(data), batch_size, replace=False)
        temp_list = []
        for i in range(index):
            cmi_data = data[i]['value']
            legal_actions = data[i]['legal_actions']
            if len(legal_actions) == 1:
                continue
            cmi_data_modified = [[action, cmi_data[1], cmi_data[2]] for action in legal_actions if action != cmi_data[0]]
            temp_list.append(cmi_data_modified)
        batch = np.array(random.sample(temp_list, batch_size))
        
    return batch


def construct_batch(data, arrange, set_size=100, K_neighbors=10):
    n = data.shape[0]
    train_index = np.random.choice(range(n), set_size, replace=False)
    test_index = np.array([i for i in range(n) if i not in train_index])

    Train_set = [data[i][train_index] for i in range(len(data))]
    Test_set = [data[i][test_index] for i in range(len(data))]

    joint_target = np.repeat([[1,0]],set_size,axis=0)
    prod_target = np.repeat([[0,1]],set_size,axis=0)
    target_train = np.concatenate((joint_target,prod_target),axis=0)
    target_train = autograd.Variable(torch.tensor(target_train).float())
    
    joint_train = sample_batch(Train_set, arrange, batch_size=set_size,sample_mode='joint')
    prod_train = sample_batch(Train_set, arrange, batch_size=set_size,sample_mode='prod_iso_kNN')
    batch_train = autograd.Variable(torch.tensor(np.concatenate((joint_train, prod_train))).float())
    
    joint_test = sample_batch(Test_set, arrange, batch_size=set_size,sample_mode='joint')
    joint_test = autograd.Variable(torch.tensor(joint_test).float())
    prod_test = sample_batch(Test_set, arrange, batch_size=set_size,sample_mode='prod_iso_kNN')
    prod_test = autograd.Variable(torch.tensor(prod_test).float())

    return batch_train, target_train, joint_test, prod_test


def estimate_cmi(model, joint_batch, prod_batch):
    gamma_joint = model(joint_batch).detach()
    gamma_prod = model(prod_batch).detach()

    sum1 = 0
    sum2 = 0

    for i in range(joint_batch.shape[0]):
        sum1 += np.log(gamma_joint[i] / (1 - gamma_joint[i]))
    for i in range(prod_batch.shape[0]):
        sum2 += gamma_prod[i] / (1 - gamma_prod[i])

    return (1/joint_batch.shape[0]) * sum1 - np.log((1/prod_batch.shape[0]) * sum2)

def train_classifer(batch_train, target_train, params, epoch, learning_rate, seed, epsilon, eval, joint_eval=[], prod_eval=[]):
    loss_e = []
    last_loss = 1000
    CMI_DV_e = []

    #Set up the model
    torch.manual_seed(seed)
    (input_size, hidden_size, num_classes, tau) = params
    model = ClassifierModel(input_size, hidden_size, num_classes, tau)    
    opt = optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(int(epoch)):
        out = model(batch_train)        
        _, pred = out.max(1)        

        loss = F.binary_cross_entropy(out, target_train) 
        loss_e.append(loss.detach().numpy())
        
        if eval:
            CMI_eval = estimate_cmi(model, joint_eval,prod_eval)
            print('epoch: ',epoch,'  ,', CMI_eval[1], ' loss: ',loss_e[-1])        
            CMI_DV_e.append(CMI_eval[1])      
        
        if abs(loss-last_loss) < epsilon and epoch > 50:
            print('epoch=',epoch)
            break

        last_loss = loss
        model.zero_grad()
        loss.backward()
        opt.step()
    if eval:    
        return model, loss_e, CMI_DV_e
    else:
        return model, loss_e
    