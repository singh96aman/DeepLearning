import numpy as np

a = np.array([1,2,3,4])
print (a)

import time
a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()
print c

print ("Vectorized version "+ str(1000*(toc-tic))+"ms")

c=0
tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
toc = time.time()

print (c)
print ("For loop "+str(1000*(toc-tic))+"ms")

A = np.array([[56, 0, 4, 68],
             [1.2,104,52,8],
             [1.8,135,99,0.9]])

print A

cal = A.sum(axis=0)
print cal

percentage = 100*A/(cal.reshape(1,4))
print percentage

a = np.random.randn(3, 3)
b = np.random.randn(3, 1)
c = a*b

print c
