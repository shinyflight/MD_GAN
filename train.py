from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import mean
from torch.autograd import Variable
import yaml
from lib import model, utils
from utils import uneye, test

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
G = model.Generator(g_input_size, g_hidden_size, g_output_size)
D = model.Discriminator(d_input_size, d_hidden_size, d_output_size)
AE = model.AutoEncoder(ae_input_size, ae_hidden_size)

# save parameter list
theta_D_gan = [D.fc1._parameters['weight'], D.fc1._parameters['bias'],
               D.out_gan._parameters['weight'], D.out_gan._parameters['bias']]
theta_D_aux = [D.out_aux._parameters['weight'], D.out_aux._parameters['bias']]
theta_G = [G.fc1._parameters['weight'], G.fc1._parameters['bias'],
           G.fc2._parameters['weight'], G.fc2._parameters['bias'],]

# define function for calculating loss function
loss_bce = torch.nn.BCELoss().cuda()
loss_nll = torch.nn.NLLLoss().cuda()

# GPU mode
if config['CUDA'] == True:
    G.cuda(), D.cuda(), AE.cuda()
    loss_bce.cuda(), loss_nll.cuda()

# define optimizers
G_solver = optim.RMSprop(theta_G, lr=lr)
D_solver = optim.RMSprop(theta_D_gan + theta_D_aux, lr=lr)
AE_solver = optim.Adam(AE.parameters(), lr=lr)


# ##### Load dataset
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
                z = Variable(z_mb, volatile=True).cuda() # inference mode
                z_y = Variable(zy_mb, volatile=True).cuda()
                # real & fake labels
                y_real = Variable(torch.ones(y.size()[0]).unsqueeze(1)).cuda()
                y_fake = Variable(torch.zeros(y.size()[0]).unsqueeze(1)).cuda()

                ## Discriminator
                for d_step in range(d_steps):
                    for p in D.parameters():  # reset requires_grad
                        p.requires_grad = True  # they are set to False below in netG update
                    D.train(True)
                    D.zero_grad()
                    # generate fake data
                    G_z = G(z, z_y)
                    X_fake = Variable(G_z.data).cuda() # volatile = False
                    # forward
                    D_real, C_real = D(X_real)  # model output
                    D_fake, C_fake = D(X_fake)
                    # calculate accuracy
                    _, pred_real = torch.max(C_real.data, 1)
                    _, pred_fake = torch.max(C_fake.data, 1)
                    total = y.size(0)#*2  # calc the number of examples
                    y_c = uneye(y, 'pred')
                    correct = torch.sum(pred_real == y_c.data)
                    #correct += torch.sum(pred_fake == y_c.data)
                    train_acc = correct/total * 100
                    # loss
                    D_real_loss = loss_bce(D_real, y_real)
                    D_fake_loss = loss_bce(D_fake, y_fake)
                    #D_real_loss = -mean(D_real + eps)  # WGAN loss
                    #D_fake_loss = mean(D_fake + eps)
                    C_real_loss = loss_nll(C_real, y_c)
                    C_fake_loss = loss_nll(C_fake, y_c)
                    DC_real_loss = D_real_loss + C_real_loss
                    DC_fake_loss = D_fake_loss + C_fake_loss
                    # backprop & update params : split real and fake loss (GAN Hack)
                    DC_real_loss.backward()
                    D_solver.step()
                    DC_fake_loss.backward()
                    D_solver.step()

                    # weight clipping
                    #for p in theta_D_gan:
                    #    p.data.clamp_(-.01, .01)

                ## Generator
                for g_step in range(g_steps):
                    for p in D.parameters():
                        p.requires_grad = False  # to avoid computation
                    G.zero_grad()
                    # generate fake data
                    z.volatile = False
                    X_fake = G(z, y)
                    # forward
                    D_real, C_real = D(X_real)  # model output
                    D_fake, C_fake = D(X_fake)
                    # loss
                    C_real_loss = loss_nll(C_real, y_c)  # cross entropy aux loss
                    C_fake_loss = loss_nll(C_fake, y_c)
                    #G_loss = -mean(D_fake + eps)  # WGAN loss
                    G_loss = loss_bce(D_fake, y_real)
                    GC_loss = G_loss + C_real_loss + C_fake_loss
                    # backprop & update params
                    GC_loss.backward()
                    G_solver.step()

            if epoch % log_every == 0:
                # calc test accuracy
                test_acc = test(X_test, y_test, D)
                D_loss = D_real_loss + D_fake_loss
                C_loss = C_real_loss + C_fake_loss
                print('epoch: %s; D: %s; G: %s; C: %s; train_acc: %.1f; test_acc: %.1f'
                      % (epoch, extract(D_loss)[0], extract(G_loss)[0], extract(C_loss)[0],train_acc, test_acc))


