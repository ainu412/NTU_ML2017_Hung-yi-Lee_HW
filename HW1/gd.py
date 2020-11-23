import csv
import os
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
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
    list_inference_time = []
    n_train = x.shape[0]

    for i in range(iteration):
        # record start time
        start = time.time() * 1000 # ms
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
        # record end time
        end = time.time() * 1000 # ms
        list_inference_time.append(end - start)
    return w, list_loss, list_inference_time

def Adagrad(x, y, w, eta, iteration, lambdaL2):
    list_loss = []
    list_inference_time = []
    n_train = x.shape[0]
    sum_grad_square = np.zeros(x.shape[1])# every xi has one wi

    for i in range(iteration):
        # record start time
        start = time.time() * 1000 # ms
        hypo = np.dot(x, w)
        diff = hypo - y
        loss = np.sum(diff ** 2) / n_train  # no regularization loss, MSE(mean square error)
        list_loss.append(loss)

        grad = np.dot(x.T, diff) / n_train + lambdaL2 * w  # regularization gradient

        sum_grad_square += grad **2
        w = w - eta * grad / np.sqrt(sum_grad_square)
        # record end time
        end = time.time() * 1000 # ms
        list_inference_time.append(end - start)
    return w, list_loss, list_inference_time


def SGD(x, y, w, eta, iteration, lambdaL2):
    list_loss = []
    list_inference_time = []
    n_train = x.shape[0]

    for i in range(iteration):
        # record start time
        start = time.time() * 1000 # ms
        hypo = np.dot(x, w)
        # record end time
        end = time.time() * 1000 # ms
        list_inference_time.append(end - start)
        diff = hypo - y
        loss = np.sum(diff ** 2) / n_train  # no regularization loss
        list_loss.append(loss)
        rand = np.random.randint(0, n_train)
        grad = x[rand] * diff[rand] + lambdaL2 * w  # regularization gradient
        w = w - eta * grad

    return w, list_loss, list_inference_time

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
ans_text.close()

trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)

# start training
## add bias to X
bias = np.array(np.ones((testX.shape[0], 1)))
testX = np.concatenate((bias, testX), axis=1)
trainX = np.concatenate((np.ones((trainX.shape[0],1)), trainX), axis=1)

w0 = np.zeros(trainX.shape[1],)
w_GD, loss_GD, inference_time_GD = GD(trainX, trainY, w0, eta=0.000001, iteration=N_ITERATION, lambdaL2=0)
w_SGD, loss_SGD, inference_time_SGD = SGD(trainX, trainY, w0, eta=0.000001, iteration=N_ITERATION, lambdaL2=0)
w_Adagrad, loss_Adagrad, inference_time_Adagrad = Adagrad(trainX, trainY, w0,eta=0.01, iteration=N_ITERATION, lambdaL2=0)

# close form
start = time.time()*1000
w_cf = inv(trainX.T.dot(trainX)).dot(trainX.T).dot(trainY)
hypo_cf = trainX.dot(w_cf)
loss_cf = np.sum((hypo_cf-trainY)**2)/len(trainX)
loss_cf = [loss_cf for i in range(N_ITERATION)]
end = time.time()*1000
avg_cf_train_inference_time_cf = (end - start) / trainX.shape[0]
avg_cf_train_inference_time_cf = [avg_cf_train_inference_time_cf for i in range(N_ITERATION)]

# output test prediction
t1=time.time_ns()
testY_GD = testX.dot(w_GD)
t2=time.time_ns()
testY_SGD = testX.dot(w_SGD)
t3=time.time_ns()
testY_Adagrad = testX.dot(w_Adagrad)
t4=time.time_ns()
testY_cf = testX.dot(w_cf)
t5=time.time_ns()
test_inference_time = []

test_inference_time.append(t5-t4) # CloseForm
test_inference_time.append(t4-t3) # Adagrad
test_inference_time.append(t2-t1) # GD
test_inference_time.append(t3-t2) # SGD


# calculate test error (MSE)
diff_GD = testY_GD-testY
mse_GD = np.sum((diff_GD)**2)/testY.shape[0]
mse_SGD = np.sum((testY_SGD-testY)**2)/testY.shape[0]
mse_Adagrad = np.sum((testY_Adagrad-testY)**2)/testY.shape[0]
mse_cf = np.sum((testY_cf-testY)**2)/testY.shape[0]

# plot
## write Adagrad prediction into csv file
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


## plot training loss with different gradient methods
plt.plot(np.arange(len(loss_cf[3:])), loss_cf[3:], 'y--', label='CloseForm')
plt.plot(np.arange(len(loss_Adagrad[3:])), loss_Adagrad[3:], 'b', label="Adagrad")
plt.plot(np.arange(len(loss_GD[3:])), loss_GD[3:], 'c', label='GD')
plt.plot(np.arange(len(loss_SGD[3:])), loss_SGD[3:], 'g', label='SGD')

plt.title('Train Process')
plt.xlabel('Iteration')
plt.ylabel('Loss Function(Quadratic)')
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), "figures/TrainLossProcess"))
plt.show()

## plot train inference time
plt.plot(np.arange(len(avg_cf_train_inference_time_cf[3:])), avg_cf_train_inference_time_cf[3:], 'y--', label='CloseForm')
plt.plot(np.arange(len(inference_time_Adagrad[3:])), inference_time_Adagrad[3:], 'b', label="Adagrad")
plt.plot(np.arange(len(inference_time_GD[3:])), inference_time_GD[3:], 'c', label='GD')
plt.plot(np.arange(len(inference_time_SGD[3:])), inference_time_SGD[3:], 'g', label='SGD')


plt.title('Train Process')
plt.xlabel('Iteration')
plt.ylabel('Training Time(millisecond)')
plt.legend()
plt.savefig("figures/TrainTimeProcess")
plt.show()

## plot test error and test inference time
fig, ax = plt.subplots()
def auto_text(rects):
    for rect in rects:
        ax.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='left', va='bottom')

labels = ['CloseForm', 'Adagrad', 'GD', 'SGD']
test_error = [mse_cf, mse_Adagrad, mse_GD, mse_SGD]
## inference time 小数点后两位四舍五入
for i in range(len(test_error)):
    origin_num = Decimal(test_error[i])
    answer_num = origin_num.quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)
    test_error[i] = answer_num

index = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots()
rect1 = ax.bar(index - width / 2, test_error, color ='r', width=width, label ='Real Result')
rect2 = ax.bar(index + width / 2, test_inference_time, color ='springgreen', width=width, label ='Inference Time(ns)')

ax.set_title('Test Result')
ax.set_xticks(ticks=index)
ax.set_xticklabels(labels)

auto_text(rect1)
auto_text(rect2)

ax.legend(loc='upper left', frameon=False)
fig.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "figures/TestErrorInferenceTime"))
plt.show()

# plot final test answer
plt.figure()
plt.plot(np.arange(len(test_inference_time)), test_inference_time, 'r', label='real result')
plt.legend(loc='upper center')

plt.subplot(141)
plt.title('CloseForm')
plt.xlabel('dataset')
plt.ylabel('pm2.5')
plt.plot(np.arange((len(testY))), testY, 'r')
plt.plot(np.arange(240), testY_cf, 'y')
plt.subplot(142)
plt.title('Adagrad')
plt.xlabel('dataset')
plt.ylabel('pm2.5')
plt.plot(np.arange((len(testY))), testY, 'r')
plt.plot(np.arange(240), testY_Adagrad, 'b')
plt.subplot(143)
plt.title('GD')
plt.xlabel('dataset')
plt.ylabel('pm2.5')
plt.plot(np.arange((len(testY))), testY, 'r')
plt.plot(np.arange(240), testY_GD, 'c')
plt.subplot(144)
plt.title('SGD')
plt.xlabel('dataset')
plt.ylabel('pm2.5')
plt.plot(np.arange((len(testY))), testY, 'r')
plt.plot(np.arange(240), testY_SGD, 'g')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "figures/TestPrediction"))
plt.show()