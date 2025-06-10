Part 2 Explanation:

2.1:
As you can see from the file gen_data.py, for each number prime, for each operation, I took the pairs of numbers a,b such that 0 <= a,b < prime. For each a,b, I calculated the result of the desired operation. I randomly split each of these training points such that 70% of them ended up in training, 20% in validation, and 10% of them in testing datasets. Based on the paper, it seemed best for much of my data to be for training. Data is findable in the data directory.

2.2:
A name such as 97add0layer1 means that whatever file associated with it was trained for the number 97, for the operation add, on the first random restart, with 1 layers. With this in mind, all the logs are findable in the logs directory, all the models in the model directory, and all graphs in the plots directory. The logs can give perhaps more information on the change in loss over time than the graphs can.

Note: Sometimes training acc is slightly lower than it should be. Due to floating point issues, sometimes loss overflows/the optimizer gets confused and loss shoots up after getting too low. If the model ends inconnveniently close to one of these spikes, testing accuracy might be low. I should probably have done some sort of checkpoint/early stopping to handle this, but considering the compute necessary to train 24(!) models, I didn't want to rerun it. Looking at the loss progression should give a good sense of how good a model is. Note this is the case for 113add0layer2test_acc.png and perhaps others.
