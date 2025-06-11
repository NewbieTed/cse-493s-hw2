*2.1*
Data generation: gen_data.py
Generated data: /data

As you can see in the file gen_data.py, for each prime number p and each operation, we took the generated pairs of numbers a,b such that 0 <= a,b < p. For each a,b, we calculated the result of the desired operation, then we randomly split each of these training points such that 70% of them ended up in training, 20% in validation, and 10% of them in testing datasets. Based on the paper, it seemed best for much of the data to be for training. Data is available in the /data directory.


*2.2*
Model checkpoints: /models directory.
A name such as 97add0layer1 refers to a model trained on p=97 for the modulo addition task, 0th random restart, with 1 feed-forward layer. [Prime][Operation][Random Restart #]layer[# layers]
Plots: /plots directory. 
The naming scheme is the same as above.
Logs: /Logs directory.
The naming scheme is the same as above.

We plot train loss over time against optimization steps for each seed and model. We additionally plot the final test accuracy of the model after training is completed. This information is included in the plots directory and also included in the appropriate log file inside of Logs/ (the test accuracy is printed at the end of the log).

Note: Sometimes training/validation accuracy is slightly lower than it should be. Due to some peculiar optimizer issues, especially when loss gets very low, loss shoots up and accuracy decreases. If the model training ends inconveniently close to one of these spikes, testing accuracy might be low. We probably should have done some sort of checkpoint/early stopping to handle this, but considering the compute necessary to train 24(!) models, we didn't want to rerun it. Looking at the loss progression should give a good sense of how good a model is.
