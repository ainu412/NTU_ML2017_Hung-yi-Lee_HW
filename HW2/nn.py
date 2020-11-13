import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import  Dense
from gp import data_process_x
from gp import data_process_y
from gp import split_n_cross_valid_data
import time

batch_size = 32
epoch_num = 300

def cal_acc(model, x, y_head, write=False):
    pred = np.squeeze(model.predict(x))
    y_round = np.around(pred)
    accuracy = np.sum(np.squeeze(y_head) == y_round) / x.shape[0]
    if write:
        df = pd.DataFrame({"id": range(1, 16282), "label": y_round})
        df.to_csv('output/nn_pred.csv')
    return accuracy

if __name__ == "__main__":
    start = time.time()
    trainData = pd.read_csv("data/train.csv")
    testData = pd.read_csv("data/test.csv")
    ans = pd.read_csv("data/correct_answer.csv")

    train_x_all = data_process_x(trainData).drop(['native_country_ Holand-Netherlands'], axis=1).values
    test_x = data_process_x(testData).values
    train_y_all = data_process_y(trainData).values
    test_y = ans['label'].values
    n = 4
    train_x, train_y, valid_x, valid_y = split_n_cross_valid_data(train_x_all, train_y_all, n)

    model = Sequential()
    model.add(Dense(units=2, activation='sigmoid', input_dim=106))
    model.add(Dense(units=2, activation='sigmoid'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # valid_accs = []
    # for i in range(n):
    #     model.fit(train_x[i], train_y[i], batch_size=batch_size, epochs=epoch_num)
    #     file_name = 'output/nn_model_valid-' + str(i) + '.h5'
    #     model.save(file_name)
    #     valid_acc = cal_acc(model, valid_x[i], valid_y[i])
    #     valid_accs.append( valid_acc )
    #
    # model.fit(train_x_all, train_y_all, batch_size=batch_size, epochs=epoch_num)
    # model.save('output/nn_model_test.h5')
    # test_acc = cal_acc(model, test_x, test_y, True)
    # print('Valid ACC:', valid_acc)
    # print('AVG Valid ACC: %.5f | Test ACC: %.5f' %  ( np.sum(valid_acc) / n  , test_acc))
    #
    # end = time.time()
    # print('Training + Test Time (s): %f' % (end-start))
    li_valid_acc = []
    for i in range(n):
        file_path = 'output/nn_model_valid-' + str(i) + '.h5'
        model.load_weights(file_path)
        valid_acc = cal_acc(model, valid_x[i], valid_y[i])
        print('Valid ACC: %.5f' % valid_acc)
        li_valid_acc.append(valid_acc)

    model.load_weights('output/nn_model_test.h5')
    test_acc = cal_acc(model, test_x, test_y)
    print('AVG Valid ACC: %.5f | Test ACC: %.5f' %( np.mean(li_valid_acc), test_acc))