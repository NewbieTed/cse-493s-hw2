Part 1

Training: train.py (logged in Part1Training.log)
Inference: inference.py (logged in Part1Inference.log)
Model Checkpoint: available in the file model.pt for the loveML.txt sanity check
How to run: To train the model on the loveML.txt dataset, simply run python train.py and the model will be run until loss is extremely low. In order to run inference, call python inference.py model.pt 50 where model.pt is the name of a model checkpoint in the local directory and 50 (or any other number) is the desired number of generated tokens (generated from scratch).

In this code, we mostly did not touch the model.py file. We implemented training in train.py and inference in inference.py. We also created a setup.py file in order to set and store hyperparameters across multiple training runs and models. We also wrote some tokenization code to encode characters as tokens and decode the tokens back to characters.
Challenges: While training on loveML.txt, we initially struggled to get the proper memorization behavior. However, once we increased batch_size from the model quickly exhibited the correct behavior. We believe that with the reduced batch size, the model’s updates were too stochastic and it struggled to optimize correctly. 

Once we increased the batch size, we noticed that the model was training very slowly and converging prematurely. To solve this challenge, we increased the learning rate to 1e-3. With the increased learning rate the model converged much more quickly and seemed to reach a better optima. On the loveML.txt dataset we noticed that the loss dropped monotonically and did not oscillate. This led us to believe that our learning rate was not too high since if it was too high the loss would oscillate. In particular, because we sample the same string over and over, a high learning rate does not affect the monotonic decrease in loss.

Finally, we got weird behavior on the loveML dataset before adjusting block_size to encompass the entire string. This makes sense because otherwise the model might see characters with incomplete context and make incorrect future predictions.
