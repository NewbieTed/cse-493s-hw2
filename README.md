**Part 2**

*2.1*

Deliverables:
- Data generation: gen_data.py
- Generated data: /data

As you can see in the file gen_data.py, for each prime number p and each operation, we took generated pairs of numbers a,b such that 0 <= a,b < p. For each a,b, we calculated the result of the desired operation, then we randomly split each of these training points such that 70% of them ended up in training, 20% in validation, and 10% of them in testing datasets. Based on the paper, it seemed best for much of the data to be for training. Data is available in the /data directory.


*2.2*

Deliverables:
- Model checkpoints: /models
  A name such as 97add0layer1 refers to a model trained on p=97 for the modulo addition task, 0th random restart, with 1 feed-forward layer.
  [Prime][Operation][Random Restart #]layer[# layers]
- Plots: /plots
  The naming scheme is the same as above.
- Logs: /logs
  The naming scheme is the same as above.

Note: Sometimes training accuracy is slightly lower than it should be. Due to floating point issues, sometimes loss underflows and as a result the optimizer gets confused and loss shoots up after getting too low. If the model training ends inconveniently close to one of these spikes, testing accuracy might be low. We should probably have done some sort of checkpoint/early stopping to handle this, but considering the compute necessary to train 24(!) models, we didn't want to rerun it. Looking at the loss progression should give a good sense of how good a model is.
