from __future__ import division
import numpy as np
import sklearn.datasets
import yaml
from copy import copy
import torch
from torch.autograd import Variable

with open('MD_GAN.yaml') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

## hyperparameters
num_fold = config['NUM_FOLD']
num_class = config['NUM_CLASS']
data_dir = config['DATA_DIR']
file_name = config['FILE_NAME']
epochs = config['EPOCHS']
batch_size = config['BATCH_SIZE']
lr = config['LEARNING_RATE']
z_dim = config['G_INPUT_SIZE']

def sample_toy(batch_size):
    while True:
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=0.25)[0]
        data = data.astype('float32')[:, [0, 2]]
        data /= 7.5  # stdev plus a little
        yield data
    # usage
    # a=sample_toy(10)
    # print(a.next())

def split_label(data, num_class):
    np.random.shuffle(data)
    X_data = np.reshape(data[:,0:-1],(-1,data.shape[1]-1))
    y_data = np.reshape(data[:, -1:], (data.shape[0]))
    y_data = y_data.astype(int)
    targets = np.array(y_data).reshape(-1)
    y_data = np.eye(num_class)[targets]
    return X_data, y_data

def split_fold(X, y, num_fold):
    num_data = X.shape[0]
    cut_num = int(1 / num_fold * num_data)
    X_fold = []
    y_fold = []
    X_remainder = []
    y_remainder = []
    for i in range(num_fold):
        cut_idx = range(cut_num*i, cut_num*(i+1))
        # X_fold
        fold_x = X[cut_idx, :]
        X_fold.append(fold_x)
        # y_fold
        fold_y = y[cut_idx, :]
        y_fold.append(fold_y)
        # remainder_fold
        res_idx = np.ones((num_data,), bool)
        res_idx[cut_idx] = False
        remainder_x = X[res_idx, :]
        remainder_y = y[res_idx, :]
        X_remainder.append(remainder_x)
        y_remainder.append(remainder_y)
    return X_fold, y_fold, X_remainder, y_remainder

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m,n])

def load_data():
    phr_feature = np.loadtxt(data_dir + file_name + '.csv',
                             delimiter=',', dtype=np.float32)
    X_data, y_data = split_label(phr_feature, num_class)
    X_ex_fold, y_ex_fold, X_ex_remain, y_ex_remain = split_fold(X_data, y_data, num_fold)

    print('\ndataset: {}'.format(file_name))
    print('total_epoch: {}; batch_size: {}; learning_rate: {}; num_fold: {}'
          .format(epochs, batch_size, lr, num_fold))

    for ex_fold in range(num_fold):
        X_test = X_ex_fold[ex_fold]
        y_test = y_ex_fold[ex_fold]
        X_in_fold, y_in_fold, X_in_remain, y_in_remain = \
            split_fold(X_ex_remain[ex_fold], y_ex_remain[ex_fold],num_fold)
        print('\n-------------------------------------------------------------------------\n')
        print('ex_fold: {}; test_size: {}'.format(ex_fold + 1, y_test.shape[0]))

        for in_fold in range(num_fold):
                # valid set
                X_valid = X_in_fold[in_fold]
                y_valid = y_in_fold[in_fold]
                # train set
                X_train = X_in_remain[in_fold]
                y_train = y_in_remain[in_fold]
                print('\nin_fold: {}; train_size: {}; valid_size: {}'
                      .format(in_fold+1, y_train.shape[0], y_valid.shape[0]))
                yield X_train, y_train, X_valid, y_valid, X_test, y_test

def load_minibatch(X_train, y_train):
    for epoch in range(epochs):
        assert np.shape(X_train)[0] == np.shape(y_train)[0]
        num_train = np.shape(X_train)[0]
        rnd_shu = np.random.permutation(num_train)
        # shuffle train set
        X_train = np.array(X_train)[rnd_shu]
        y_train = np.array(y_train)[rnd_shu]
        for start_idx in range(0, num_train, batch_size):
            start = start_idx
            end = start_idx + batch_size
            if end > num_train:
                end = num_train
            # mini-batch
            X_mb = torch.FloatTensor(X_train[start:end])
            y_mb = torch.FloatTensor(y_train[start:end])
            z_mb = torch.FloatTensor(sample_z(end - start, z_dim))
            zy_mb = copy(y_mb)
            yield X_mb, y_mb, z_mb, zy_mb

def uneye(y, type):
    # process y
    _, y = torch.max(y, 1)
    if type == 'model':
        y = y.type(torch.FloatTensor).cuda()
        y = y.unsqueeze(1)
    elif type == 'pred':
        y = y.type(torch.LongTensor).cuda().squeeze()
    else:
        raise TypeError
    return y

def test(X_test, y_test, D):
    # preprocess
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    X_test = Variable(X_test, volatile=True)
    y_test = Variable(y_test)
    if config['CUDA'] == True:
        X_test = X_test.cuda()
        y_test = y_test.cuda()
    # prediction
    _, C_test = D(X_test)
    #C_test.detach()
    _, pred_test = torch.max(C_test.data, 1)
    # label
    total = y_test.size(0)  # calc the number of examples
    y_t = uneye(y_test, 'pred')
    # calculate accuracy
    correct = torch.sum(pred_test == y_t.data)
    test_acc = correct / total * 100
    return test_acc

if __name__ == '__main__':
    a=load_data()
    for i in range(8):
        X_train, y_train, X_valid, y_valid, X_test, y_test = next(a)
        print(X_train[0])