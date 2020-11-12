import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from gp import data_process_x, split_valid_data
from gp import data_process_y
from gp import sigmoid
from gp import _shuffle
from gp import split_n_cross_valid_data
import os

l_rate = 0.001
batch_size = 32
epoch_num = 300

def cal_acc(x, y_head, w):
    y = sigmoid( np.dot(x, w) )
    acc = np.sum(np.around(y) == y_head) / x.shape[0]
    return acc

def train_process(x, y):
    w = np.zeros(x.shape[1])
    list_loss = []
    batch_num = x.shape[0] // batch_size
    for _ in range(epoch_num):
        x, y = _shuffle(x, y)
        epoch_loss = 0.0
        for i in range(batch_num):
            batch_x = x[i*batch_size : (i + 1)*batch_size]
            batch_y = np.squeeze(y[i * batch_size : (i + 1) * batch_size])

            pred = sigmoid( np.dot(batch_x, w) )
            cross_entropy = (-np.dot(batch_y, np.log(pred)) - np.dot(1 - np.squeeze(batch_y.T), np.log(1 - pred))) / batch_size
            grad = np.sum(-batch_x * (batch_y - pred).reshape((batch_size, 1)), axis=0)
            epoch_loss += cross_entropy

            w = w - l_rate * grad

        list_loss.append(epoch_loss)

    plt.plot(np.arange(len(list_loss)), list_loss)
    plt.title("Train Process")
    plt.xlabel("epoch_num")
    plt.ylabel("Loss Function (Cross Entropy)")
    plt.savefig("output/TrainProcess")
    plt.show()

    return w


if __name__ == "__main__":
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    ans = pd.read_csv('data/correct_answer.csv')

    train_x_all = data_process_x(train).drop('native_country_ Holand-Netherlands', axis=1).values # train比test多一个属性,转为one-hot时就会多一列,导致mu1/mu2多一列
    train_y_all = data_process_y(train).values
    test_x = data_process_x(test).values
    test_y = ans['label'].values
    # add bias
    test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1)
    train_x_all = np.concatenate((np.ones((train_x_all.shape[0], 1)),train_x_all), axis=1)

    # train_x, train_y, valid_x, valid_y = split_valid_data(train_x_all, train_y_all, 0.1)
    # w = train_process(train_x, train_y)
    # valid_acc = cal_acc(valid_x, valid_y, w)
    # w = train_process(train_x_all, train_y_all)
    # test_acc = cal_acc(test_x, test_y, w)
    # print('Valid ACC: %.5f | Test ACC: %.5f' % (valid_acc, test_acc))

    # n-cross validation
    n = 4
    train_x, train_y, valid_x, valid_y = split_n_cross_valid_data(train_x_all, train_y_all, n)
    valid_accs = np.zeros(n,)
    test_accs = np.zeros(n,)

    for i in range(n):
        w = train_process(train_x[i], train_y[i])
        valid_acc = cal_acc(valid_x[i], valid_y[i], w)
        test_acc = cal_acc(test_x, test_y, w)
        print('Valid ACC: %.5f | Test ACC: %.5f' % (valid_acc, test_acc))
        valid_accs[i] = valid_acc
        test_accs[i] = test_acc

    avg_valid_acc = np.sum(valid_accs) / n
    avg_test_acc = np.sum(test_accs) / n
    print('AVG Valid ACC: %.5f | AVG Test ACC: %.5f' % (avg_valid_acc, avg_test_acc))

