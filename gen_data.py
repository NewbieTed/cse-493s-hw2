import random as r
import sys

r.seed(1)
max = 1000

train_size = 10000
val_size = 2000
test_size = 2000

log_file = open("pt2train.txt", "w")
sys.stdout = log_file

for _ in range(train_size):
    i = r.randint(0,1)
    if i == 0:
        num = 97
    else:
        num = 113
    i = r.randint(0,2)
    a = r.randint(0,max)
    b = r.randint(0,max)
    if i == 0:
        c = (a + b) % num
        print(a, " + ", b, " = ", c)
    elif i == 1:
        c = (a - b) % num
        print(a, " - ", b, " = ", c)
    else:
        c = (a * b) % num
        print(a, " * ", b, " = ", c)
        
log_file = open("pt2val.txt", "w")
sys.stdout = log_file

for _ in range(val_size):
    i = r.randint(0,1)
    if i == 0:
        num = 97
    else:
        num = 113
    i = r.randint(0,2)
    a = r.randint(0,max)
    b = r.randint(0,max)
    if i == 0:
        c = (a + b) % num
        print(a, " + ", b, " = ", c)
    elif i == 1:
        c = (a - b) % num
        print(a, " - ", b, " = ", c)
    else:
        c = (a * b) % num
        print(a, " * ", b, " = ", c)

log_file = open("pt2test.txt", "w")
sys.stdout = log_file

for _ in range(test_size):
    i = r.randint(0,1)
    if i == 0:
        num = 97
    else:
        num = 113
    i = r.randint(0,2)
    a = r.randint(0,max)
    b = r.randint(0,max)
    if i == 0:
        c = (a + b) % num
        print(a, " + ", b, " = ", c)
    elif i == 1:
        c = (a - b) % num
        print(a, " - ", b, " = ", c)
    else:
        c = (a * b) % num
        print(a, " * ", b, " = ", c)

log_file.close()
