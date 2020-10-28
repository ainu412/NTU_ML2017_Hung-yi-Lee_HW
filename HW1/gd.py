import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

N_POLLUTANT = 18
N_HOUR_PER_DAY = 24
N_MONTH_PER_YEAR = 12
N_TRAIN_DAY_PER_MONTH = 20
N_INPUT_X_HOUR = 9
N_TRAIN_Y_HOUR = 1
INDEX_PM_2_5 = 9
INDEX_TRAIN_Y_HOUR =N_INPUT_X_HOUR
N_TRAIN_HOUR_PER_MONTH = N_HOUR_PER_DAY * N_TRAIN_DAY_PER_MONTH #480

N_ITERATION = 200

def GD(x, y, w, eta, iteration, lambdaL2):
    list_loss = []
    n_train = x.shape[0]

    for i in range(iteration):
        hypo = np.dot(x, w)
        diff = hypo - y
        # prevent overflow
        for dif in range(len(diff)):
            if diff[dif] > np.sqrt(sys.maxsize):
                diff[dif] = np.sqrt(sys.maxsize)

        loss = np.sum(diff**2) / n_train # no regularization loss
        list_loss.append(loss)
        grad = np.dot(x.T, diff) / n_train + lambdaL2 * w# regularization gradient
        w = w - eta * grad

    return w, list_loss

def Adagrad(x, y, w, eta, iteration, lambdaL2):
    list_loss = []
    n_train = x.shape[0]
    sum_grad_square = np.zeros(x.shape[1])# every xi has one wi

    for i in range(iteration):
        hypo = np.dot(x, w)
        diff = hypo - y
        loss = np.sum(diff ** 2) / n_train  # no regularization loss, MSE(mean square error)
        list_loss.append(loss)

        grad = np.dot(x.T, diff) / n_train + lambdaL2 * w  # regularization gradient

        sum_grad_square += grad **2
        w = w - eta * grad / np.sqrt(sum_grad_square)

    return w, list_loss


def SGD(x, y, w, eta, iteration, lambdaL2):
    list_loss = []
    n_train = x.shape[0]

    for i in range(iteration):
        hypo = np.dot(x, w)
        diff = hypo - y
        loss = np.sum(diff ** 2) / n_train  # no regularization loss
        list_loss.append(loss)
        rand = np.random.randint(0, n_train)
        grad = x[rand] * diff[rand] + lambdaL2 * w  # regularization gradient
        w = w - eta * grad

    return w, list_loss

# parse training data
## get pollutant table
train_src_text = open('data/train.csv', 'r', encoding='big5')
row = csv.reader(train_src_text, delimiter=',')
n_row = 0
pollutant = []
for p in range(N_POLLUTANT):
    pollutant.append([])
fir_line = True
for r in row:
    # ignore top header
    if fir_line:
        fir_line = False
        continue
    # ignore left header
    for column in range(3,27):
        if r[column] == 'NR':
            pollutant[n_row % N_POLLUTANT].append(float(0))
        else:
            pollutant[n_row % N_POLLUTANT].append(float(r[column]))
    n_row = n_row + 1
train_src_text.close()


## get trainX and trainY
START_HOUR_MAX = N_HOUR_PER_DAY * N_TRAIN_DAY_PER_MONTH - (N_INPUT_X_HOUR + N_TRAIN_Y_HOUR - 1)
trainX = []
trainY = []
for m in range(12):
    for start_hour_in_month in range(471):
        trainX.append([])
        for p in range(18):
            for h in range(9):
                trainX[m*START_HOUR_MAX+start_hour_in_month].append\
                    (pollutant[p][m * N_TRAIN_HOUR_PER_MONTH + start_hour_in_month + h])
        trainY.append(pollutant[INDEX_PM_2_5][m * N_TRAIN_HOUR_PER_MONTH + start_hour_in_month + INDEX_TRAIN_Y_HOUR])

# parse test data
test_text = open('data/test.csv', 'r', encoding='big5')
row = csv.reader(test_text, delimiter=',')
testX = []

n_row = 0
for r in row:
    if n_row % 18 == 0:
        testX.append([])
    for column in range(2, 2 + N_INPUT_X_HOUR):
        if r[column] == 'NR':
            testX[n_row // 18].append(float(0))
        else:
            testX[n_row // 18].append(float(r[column]))
    n_row = n_row + 1
test_text.close()

# parse test answer
ans_text = open('data/ans.csv', 'r', encoding='big5')
row = csv.reader(ans_text, delimiter=',')
n_row = 0
testY = []
for r in row:
    testY.append(r[1])
testY = testY[1:] # ignore top header
testY = list(map(int,testY))
testY = np.array(testY)
print(testY)
ans_text.close()

# start training
## add bias to X
trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)

bias = np.array(np.ones((testX.shape[0], 1)))
testX = np.concatenate((bias, testX), axis=1)
trainX = np.concatenate((np.ones((trainX.shape[0],1)), trainX), axis=1)

w0=np.zeros(trainX.shape[1])
w_GD, loss_GD = GD(trainX, trainY, w0, eta=0.000001, iteration=N_ITERATION, lambdaL2=0)
w_SGD, loss_SGD = SGD(trainX, trainY, w0, eta=0.000001, iteration=N_ITERATION, lambdaL2=0)
w_Adagrad, loss_Adagrad = Adagrad(trainX, trainY, w0,eta=0.01, iteration=N_ITERATION, lambdaL2=0)

# close form
w_cf = inv(trainX.T.dot(trainX)).dot(trainX.T).dot(trainY)
hypo_cf = trainX.dot(w_cf)
loss_cf = np.sum((hypo_cf-trainY)**2)/len(trainX)
loss_cf = [loss_cf for i in range(N_ITERATION)]

# output test prediction
testY_GD = testX.dot(w_GD)
testY_SGD = testX.dot(w_SGD)
testY_Adagrad = testX.dot(w_Adagrad)
testY_cf = testX.dot(w_cf)

# calculate test error (MSE)
diff_GD = testY_GD-testY
mse_GD = np.sum((diff_GD)**2)/testY.shape[0]
mse_SGD = np.sum((testY_SGD-testY)**2)/testY.shape[0]
mse_Adagrad = np.sum((testY_Adagrad-testY)**2)/testY.shape[0]
mse_cf = np.sum((testY_cf-testY)**2)/testY.shape[0]
print('error_GD: %f | error_SGD: %f | error_Adagrad: %f | error_cf: %f | '
      % (mse_GD, mse_SGD, mse_Adagrad, mse_cf))

# write Adagrad prediction into csv file
ans = []
for month in range(len(testX)):
    ans.append(["id_" + str(month)])
    a = np.dot(w_Adagrad, testX[month])
    ans[month].append(a)

filename = "result/predict.csv"
text = open(filename, "w+")
hour = csv.writer(text, delimiter=',', lineterminator='\n')
hour.writerow(["id", "value"])
for month in range(len(ans)):
    hour.writerow(ans[month])
text.close()


# plot training data with different gradient methods
plt.plot(np.arange(len(loss_Adagrad[3:])), loss_Adagrad[3:], 'b', label="Adagrad")
plt.plot(np.arange(len(loss_SGD[3:])), loss_SGD[3:], 'g', label='SGD')
plt.plot(np.arange(len(loss_GD[3:])), loss_GD[3:], 'r', label='GD')
plt.plot(np.arange(len(loss_cf[3:])), loss_cf[3:], 'y--', label='close-form')

plt.title('Train Process')
plt.xlabel('Iteration')
plt.ylabel('Loss Function(Quadratic)')
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), "figures/TrainLossProcess"))
plt.show()

# plot final test answer
plt.figure()
plt.subplot(141)
plt.title('CloseForm')
plt.xlabel('dataset')
plt.ylabel('pm2.5')
plt.plot(np.arange((len(testY))), testY, 'r')
plt.plot(np.arange(240), testY_cf, 'c')
plt.subplot(142)
plt.title('Adagrad')
plt.xlabel('dataset')
plt.ylabel('pm2.5')
plt.plot(np.arange((len(testY))), testY, 'r')
plt.plot(np.arange(240), testY_Adagrad, 'g')
plt.subplot(143)
plt.title('SGD')
plt.xlabel('dataset')
plt.ylabel('pm2.5')
plt.plot(np.arange((len(testY))), testY, 'r')
plt.plot(np.arange(240), testY_SGD, 'b')
plt.subplot(144)
plt.title('GD')
plt.xlabel('dataset')
plt.ylabel('pm2.5')
plt.plot(np.arange((len(testY))), testY, 'r')
plt.plot(np.arange(240), testY_GD, 'y')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "figures/Compare"))
plt.show()