from math import floor

import numpy as np
import pandas as pd
attribute_num = 106 # exclude Holand_Netherlands

def data_process_x(raw):

    #sex 只有两个属性 先drop之后处理
    if "income" in raw.columns:
        data = raw.drop(["sex", 'income'], axis=1)
    else:
        data = raw.drop(["sex"], axis=1)
    object_col = [col for col in data.columns if data[col].dtypes == "object"] #读取非数字的column
    non_object_col = [x for x in data.columns if x not in object_col] #数字的column

    object_data = data[object_col]
    non_object_data = data[non_object_col]
    #insert set into nonobject data with male = 0 and female = 1
    non_object_data.insert(0, column="sex", value=(raw["sex"] == " Female").astype(np.int))
    #set every element in object rows as an attribute
    object_oh = pd.get_dummies(object_data)

    data = pd.concat([non_object_data, object_oh], axis=1)
    x = data.astype("int64")
    # z-score normalize
    x = (x - x.mean()) / x.std()

    return x

# 转换y: >=50K -> 1 , <50K -> -
def data_process_y(raw):
    y = pd.DataFrame((raw['income']==' >50K').astype(np.int), columns=['income'])
    y = np.squeeze(y)
    return y

def sigmoid(z):
    y = 1 / (1 + np.exp(-z))
    return np.clip(y, 1e-8, 1-1e-8)

def _shuffle(x,y):
    # 生成index
    rand_index = np.arange(x.shape[0])
    np.random.shuffle(rand_index)
    return x[rand_index], y[rand_index]

def split_valid_data(x, y, valid_percentage):
    x,y = _shuffle(x,y)
    valid_num = int(floor(x.shape[0] * valid_percentage))
    valid_x = x[:valid_num]
    valid_y = y[:valid_num]
    train_x = x[valid_num:]
    train_y = y[valid_num:]
    return train_x, train_y, valid_x, valid_y

def split_n_cross_valid_data(x, y, n):
    x,y = _shuffle(x,y)
    part = x.shape[0] // n
    x = x[:part*n]
    y = y[:part*n]
    train_x = []
    train_y = []
    valid_x = []
    valid_y = []
    for i in range(n):
        valid_x.append( x[i * part:(i + 1) * part] )
        valid_y.append( y[i * part:(i + 1) * part] )

        train_x.append(np.delete(x, range(i * part,(i + 1) * part), axis=0))
        train_y.append(np.delete(y, range(i * part,(i + 1) * part), axis=0))

    return train_x, train_y, valid_x, valid_y

def cal_acc(x, y_head, mu1, mu2, shared_sigma, N1, N2):
    inv_sigma = np.linalg.inv(shared_sigma)
    w = np.dot((mu1 - mu2).T, inv_sigma)
    b = -0.5 * np.dot(np.dot(mu1.T, inv_sigma), mu1) + 0.5 * np.dot(np.dot(mu2.T, inv_sigma), mu2) + np.log(N1 / N2)
    z = np.dot(w, x.T) + b
    y = sigmoid(z)
    accuracy = np.squeeze(y_head) == np.around(y)
    average_acc = np.sum(accuracy) / float(x.shape[0])
    return average_acc, y

def train_process(x,y):
    N = x.shape[0]
    mu1 = np.zeros((attribute_num,))
    mu2 = np.zeros((attribute_num,))
    N1 = 0
    N2 = 0
    for i in range(N):
        if y[i] == 1:
            mu1 += x[i]
            N1 += 1
        else:
            mu2 += x[i]
            N2 += 1
    mu1 /= N1
    mu2 /= N2

    sigma1 = np.zeros((attribute_num,attribute_num))
    sigma2 = np.zeros((attribute_num,attribute_num))
    for i in range(N):
        if y[i] == 1:
          sigma1 += np.dot(np.transpose([x[i] - mu1]), [x[i] - mu1])
        else:
          sigma2 += np.dot(np.transpose([x[i] - mu2]), [x[i] - mu2])
    sigma1 = sigma1/N1
    sigma2 = sigma2/N2
    common_sigma = sigma1 * N1/N + sigma2 * N2/N
    return mu1, mu2, common_sigma, N1, N2

if __name__=='__main__':
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    ans = pd.read_csv('data/correct_answer.csv')

    train_x = data_process_x(train).drop('native_country_ Holand-Netherlands', axis=1).values # train比test多一个属性,转为one-hot时就会多一列,导致mu1/mu2多一列
    train_y = data_process_y(train).values
    test_x = data_process_x(test).values
    test_y = ans['label'].values

    # train_x, train_y, valid_x, valid_y = split_valid_data(train_x, train_y, 0.2)
    # mu1, mu2, common_sigma, N1, N2 = train_process(train_x, train_y)
    # valid_acc, _ = cal_acc(valid_x, valid_y, mu1, mu2, common_sigma, N1, N2)
    # test_acc, test_pred = cal_acc(test_x, test_y, mu1, mu2, common_sigma, N1, N2)
    # print('Valid ACC: %.5f | Test ACC: %.5f' % (valid_acc, test_acc))

    # n-cross validation
    n = 4
    train_x, train_y, valid_x, valid_y = split_n_cross_valid_data(train_x, train_y, n)
    valid_accs = []
    test_accs = []
    for i in range(n):
        mu1, mu2, common_sigma, N1, N2 = train_process(train_x[i], train_y[i])
        valid_acc, _ = cal_acc(valid_x[i], valid_y[i], mu1, mu2, common_sigma, N1, N2)
        test_acc, test_pred = cal_acc(test_x, test_y, mu1, mu2, common_sigma, N1, N2)
        print('Valid ACC: %.5f | Test ACC: %.5f' % (valid_acc, test_acc))
        valid_accs.append(valid_acc)
        test_accs.append(test_acc)

    avg_valid_acc = np.sum(valid_accs) / n
    avg_test_acc = np.sum(test_accs) / n
    print('AVG Valid ACC: %.5f | AVG Test ACC: %.5f' % (avg_valid_acc, avg_test_acc))

    test_pred = pd.DataFrame({'id':np.arange(1,1+test_x.shape[0]), 'label':test_pred})
    test_pred.to_csv('output/gp_pred.csv')

