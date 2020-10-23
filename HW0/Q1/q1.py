# recommend python numpy
import numpy as np

# read txt
with open('data/matrixA.txt') as content:
    content = content.read().split(',')
    matrixA = list(map(int, content))

with open('data/matrixB.txt') as content:
    content = content.readlines()
    content = [line.strip() for line in content]
    content = [line.split(',') for line in content]
    matrixB = [list(map(int, line)) for line in content]

A = np.array(matrixA)
B = np.array(matrixB)
res = A.dot(B)
print('before sort: %d' % res)
res = np.sort(res)

print('after sort: %d' % res)
np.savetxt('ans_one.txt', res, fmt=%d)