Part 2.4 Explanation:

For our ablation experiment, we chose to test the effect of batch_size. We tested batch sizes ranging from 8-1024, doubling at each interval for a total of 8 tests. I expected that the higher batch size would cause the test_loss to decay to close to zero more quickly than smaller batches due to simply training on more data. However we saw the opposite. Unurprisingly, sometimes the model does not see enough samples for even train_loss to decay enough at very small batch_sizes (that is what the nonsensical -100 value indicates). However when train_loss does converge, a lower batch-size causes test_loss to converge more quickly relatively speaking. This might be because of the fact that test_loss stays more closely coupled to train_loss throughout the training time due to the model not-overtraining/memorizing the training data. It could also be simply due to random change -- we sometimes observed that the random starting seed could have major effects on model and validation convergence.

Plot of the relation available in better_bar_plot.png, all the other famililar logs, plots, and models available in their respective directories.


