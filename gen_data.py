import random as r
import sys

r.seed(1)
max = 1000

train_size = 10000
val_size = 2000
test_size = 2000

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
            a = r.randint(1,max)
            b = r.randint(1,max)
            if j > 2:
                div = 113
            else:
                div = 97
            if j % 3 == 0:
                c = (a + b) % div
                print(a, " + ", b, " = ", c)
            elif j % 3 == 1:
                c = (a - b) % div
                print(a, " - ", b, " = ", c)
            else:
                c = (a / b) % div 
                print(a, " / ", b, " = ", c)
        log_file.close()
            
        


# log_file = open("97+train.txt", "w")
# sys.stdout = log_file


# for _ in range(train_size):
#     num = 97
#     a = r.randint(0,max)
#     b = r.randint(0,max)
#     c = (a + b) % num
#     print(a, " + ", b, " = ", c)
#     if i == 0:
#     elif i == 1:
#         c = (a - b) % num
#         print(a, " - ", b, " = ", c)
#     else:
#         c = (a * b) % num
#         print(a, " * ", b, " = ", c)
        
# log_file = open("2val.txt", "w")
# sys.stdout = log_file

# for _ in range(val_size):
#     i = r.randint(0,1)
#     if i == 0:
#         num = 97
#     else:
#         num = 113
#     i = r.randint(0,2)
#     a = r.randint(0,max)
#     b = r.randint(0,max)
#     if i == 0:
#         c = (a + b) % num
#         print(a, " + ", b, " = ", c)
#     elif i == 1:
#         c = (a - b) % num
#         print(a, " - ", b, " = ", c)
#     else:
#         c = (a * b) % num
#         print(a, " * ", b, " = ", c)

# log_file = open("pt2test.txt", "w")
# sys.stdout = log_file

# for _ in range(test_size):
#     i = r.randint(0,1)
#     if i == 0:
#         num = 97
#     else:
#         num = 113
#     i = r.randint(0,2)
#     a = r.randint(0,max)
#     b = r.randint(0,max)
#     if i == 0:
#         c = (a + b) % num
#         print(a, " + ", b, " = ", c)
#     elif i == 1:
#         c = (a - b) % num
#         print(a, " - ", b, " = ", c)
#     else:
#         c = (a * b) % num
#         print(a, " * ", b, " = ", c)

# log_file.close()
