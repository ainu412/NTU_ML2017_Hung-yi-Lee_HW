import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gp import data_process_x, split_valid_data
from gp import data_process_y
from gp import sigmoid
from gp import _shuffle
from gp import split_n_cross_valid_data

l_rate = 0.001
batch_size = 32
epoch_num = 300
n_cross_valid = True

def cal_acc(x, y_head, w, write=False):
    y = sigmoid( np.dot(x, w) )
    y_round = np.around(y)
    acc = np.sum(y_round == y_head) / x.shape[0]
    if write:
        df = pd.DataFrame({'id': range(1,16282), 'label':y_round})
        df.to_csv('output/lr_pred.csv')
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
    start = time.time()
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

    if not n_cross_valid:
        train_x, train_y, valid_x, valid_y = split_valid_data(train_x_all, train_y_all, 0.1)
        w = train_process(train_x, train_y)
        valid_acc = cal_acc(valid_x, valid_y, w)
        w = train_process(train_x_all, train_y_all)
        test_acc = cal_acc(test_x, test_y, w)
        print('Valid ACC: %.5f | Test ACC: %.5f' % (valid_acc, test_acc))

    # n-cross validation
    if n_cross_valid:
        n = 4
        train_x, train_y, valid_x, valid_y = split_n_cross_valid_data(train_x_all, train_y_all, n)
        li_valid_acc = []
        test_accs = np.zeros(n,)

        for i in range(n):
            w = train_process(train_x[i], train_y[i])
            valid_acc = cal_acc(valid_x[i], valid_y[i], w)

            print('Valid ACC: %.5f' % valid_acc)
            li_valid_acc.append(valid_acc)

        w = train_process(train_x_all, train_y_all)
        test_acc = cal_acc(test_x, test_y, w, True)
        print('AVG Valid ACC: %.5f | Test ACC: %.5f' % (np.mean(li_valid_acc), test_acc))

    end = time.time()
    print('Training + Test Time (s): %f' % (end - start))
