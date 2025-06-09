import random as r
import sys
import itertools
import os

# r.seed(1)
# max = 1000

train_prop = 0.7
val_prop = 0.2
test_prop = 0.1

names = ["97add", "97sub", "97div", "113add", "113sub", "113div"]

def make_map(num):
    dic = {}
    for i in range(num):
        nums = [j for _ in range(num)]
        dic[i] = nums
    return dict 

def math_print(pair, j, prime):
    a = pair[0]
    b = pair[1]
    if j % 3 == 0:
        c = (a + b) % prime
        print("", a, "+", b, "=", c)
    elif j % 3 == 1:
        c = (a - b) % prime
        print("", a, "-", b, "=", c)
    else:
        if b == 0:
            return
            # b = 1
        invb = pow(b, -1, mod=prime)
        c = (a * invb) % prime
        print("", a, "/", b, "=", c)

for j in range(len(names)):
        prime = int(names[j][:2])
        pairs = list(itertools.product(range(prime), repeat=2))
        r.shuffle(pairs)


        try:
            os.remove(names[j] + "train" + ".txt")
            os.remove(names[j] + "val" + ".txt")
            os.remove(names[j] + "test" + ".txt")
        except:
            pass

        for idx, pair in enumerate(pairs):
            if idx < (train_prop * len(pairs)):
                file_name = names[j] + "train" + ".txt"
                log_file = open(file_name, "a")
                sys.stdout = log_file

            elif idx < (train_prop + val_prop) * len(pairs):
                file_name = names[j] + "val" + ".txt"
                log_file = open(file_name, "a")
                sys.stdout = log_file
            else:
                file_name = names[j] + "test" + ".txt"
                log_file = open(file_name, "a")
                sys.stdout = log_file
            math_print(pair, j, prime)
