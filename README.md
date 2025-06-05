Part 1 and 1.5 explanation:

The log of the training of the model on the file loveML.txt which contains the string "I love machine learning" is available in Part1Training.log, and the output itself is available in Part1Inference.log. The model checkpoint is available in the file loveML.pt

In this code, I mostly did not touch the model.py file. I implemented training in train.py and inference in inference.py. I also added a setup.py file to control things like hyperparameters in order to make it easier to control those across multiple trainings and multiple models. There is also some boilerplate encoding/decoding code that did not seem suitable to be elsewhere in my opinion.

Challenges: I at first did not realize when training on loveML.txt that in order to get the proper behavior, we needed to set batch_size precisely so that the model could memorize th whole dataset. Once I realized this, it became much simpler to implement