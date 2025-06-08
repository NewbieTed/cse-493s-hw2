import random as r
import sys

r.seed(1)
# max = 1000

train_size = 1000000
val_size = 200000
test_size = 200000

name = ["97add", "97sub", "97div", "113add", "113sub", "113div"]
for j in range(len(name)):
    for i in range(3):
        if i == 0:
            num = train_size
            file_name = name[j] + "train" + ".txt"
        elif i == 1:
            num = val_size
            file_name = name[j] + "val" + ".txt"
        else:
            num = test_size
            file_name = name[j] + "test" + ".txt"

        log_file = open(file_name, "w")
        for _ in range(num):
            sys.stdout = log_file
            a = r.randint(100,999)
            b = r.randint(100,999)
            if j > 2:
                div = 113
            else:
                div = 97
            if j % 3 == 0:
                c = (a + b) % div
                print(a, "+", b, "=", c)
            elif j % 3 == 1:
                c = (a - b) % div
                print(a, "-", b, "=", c)
            else:
                c = (a // b) % div 
                print(a, "/", b, "=", c)
        log_file.close()
