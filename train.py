from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import mean
from torch.autograd import Variable
import yaml
from lib import model, utils
from utils import uneye

with open("MD_GAN.yaml") as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# to print easily
def extract(v):
    return v.data.storage().tolist()

# #####  Hyper-parameters
## Set GPU ID
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = config['GPU']
## Learning params
lr = float(config['LEARNING_RATE'])
batch_size = config['BATCH_SIZE']
num_epoch = config['EPOCHS']
d_steps = config['D_STEPS']
g_steps = config['G_STEPS']
eps = float(config['epsilon'])
log_every = config['LOG_EVERY']
num_fold = config['NUM_FOLD']
## Model params
# generator
g_input_size = config['G_INPUT_SIZE']  # random noise (z)
g_hidden_size = config['G_HIDDEN_SIZE']  # generator complexity
g_output_size = config['G_OUTPUT_SIZE']  # size of generated output vector
# discriminator
d_input_size = config['D_INPUT_SIZE']
d_hidden_size = config['D_HIDDEN_SIZE']  # discriminator complexity
d_output_size = config['D_OUTPUT_SIZE']
# autoencoder
ae_input_size = config['AE_INPUT_SIZE']
ae_hidden_size = config['AE_HIDDEN_SIZE']


# ##### define the MDGAN model
G = model.Generator(g_input_size, g_hidden_size, g_output_size).cuda()
D = model.Discriminator(d_input_size, d_hidden_size, d_output_size).cuda()
AE = model.AutoEncoder(ae_input_size, ae_hidden_size).cuda()
# define function for calculating loss function
BCE = nn.CrossEntropyLoss()
# define optimizers
G_solver = optim.RMSprop(G.parameters(), lr=lr)
D_solver = optim.RMSprop(D.parameters(), lr=lr)
AE_solver = optim.Adam(AE.parameters(), lr=lr)


# ##### Load dataset
# real & fake labels
real_label = Variable(torch.ones(batch_size)).cuda()
fake_label = Variable(torch.zeros(batch_size)).cuda()
# define dataloader
load_data = utils.load_data()
# ##### train loop
for ex_fold in range(num_fold):
    for in_fold in range(num_fold):
        X_train, y_train, X_valid, y_valid, X_test, y_test = next(load_data)
        load_minibatch = utils.load_minibatch(X_train, y_train)
        num_batch = int(np.ceil(np.shape(X_train)[0]/batch_size))
        for epoch in range(num_epoch):
            for batch in range(num_batch):
                # load data batch
                x_mb, y_mb, z_mb, zy_mb = next(load_minibatch)
                X_real = Variable(x_mb).cuda()  # input features of real data
                y = Variable(y_mb).cuda()  # class targets of real data
                z = Variable(z_mb).cuda()
                z_y = Variable(zy_mb).cuda()
                ## Discriminator
                for d_step in range(d_steps):
                    D.zero_grad()
                    # generate fake data
                    X_fake = G(z, z_y)
                    # forward
                    D_real, C_real = D(X_real)  # model output
                    D_fake, C_fake = D(X_fake)
                    # calculate accuracy
                    _, pred = torch.max(C_real.data, 1)
                    total = y.size(0)  # calc the number of examples
                    y_ = uneye(y, 'pred')
                    correct = torch.sum(pred == y_.data)
                    acc = correct/total * 100
                    # loss
                    C_loss = BCE(C_real, y_) + BCE(C_fake, y_)  # cross entropy aux loss
                    D_loss = mean(D_real + eps) - mean(D_fake + eps)  # WGAN loss
                    DC_loss = -(0.01*D_loss + C_loss)
                    # backprop & update params
                    DC_loss.backward()
                    D_solver.step()
                    # weight clipping
                    for p in D.parameters():
                        p.data.clamp_(-.01, .01)
                    ## Generator
                for g_step in range(g_steps):
                    G.zero_grad()
                    # generate fake data
                    X_fake = G(z, y)
                    # forward
                    D_real, C_real = D(X_real)  # model output
                    D_fake, C_fake = D(X_fake)
                    # loss
                    C_loss = BCE(C_real, y_) + BCE(C_fake, y_)  # cross entropy aux loss
                    G_loss = mean(D_fake + eps)
                    GC_loss = -(0.01*G_loss + C_loss)
                    # backprop & update params
                    GC_loss.backward()
                    G_solver.step()

            if epoch % log_every == 0:
                print('epoch: %s; D: %s; G: %s; C: %s; train_acc: %.1f' % (epoch, extract(D_loss)[0], extract(G_loss)[0], extract(C_loss)[0],acc))