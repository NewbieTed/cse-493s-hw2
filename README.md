Part 2.3

Plot of the training curves and test accuracy: Available in the plots directory

Model checkpoints: Available in the model directory

Inference instructions: As before, run `python inference.py model.pt X` where model.pt is the path to a model checkpoint and X is the maximum number of desired tokens. The sequence will be generated from scratch (0 context).

As you can see, the model somewhat groks. It achieves partial validation accuracy and then jumps after some more runs. This is done with a 50-25-25 train, validation, test data split which is different from the 2.2 data. This is done because we were able to get more ‘grokkish’ behavior with this training split. We might have been able to get something that matched the figure from the paper better if we adjusted our optimizer, accuracy measurement, and data splits.


