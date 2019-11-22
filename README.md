# medical-bert
This repo is for collecting all the code in one place for the task of 
predicting hospital readmission from clinical text.

## Application Structure
The application is organised as a python package with 4 components:
* Classifier
* Dataloader
* Trainer
* Evaluator

The classifier defines the model and optimizer, as well as the underlying
base model (i.e. we're using a BertForSequenceClassification model).

The datareader helps load in data into a tensordataset. If it has done so previously it laods this from a cached file.

The trainer is effectively the model optimisation loop that we had extracted out the thesis code.

The evaluator reads in data and every model checkpoint and outputs a summary of scores as well as the model output for further evaluation.

There is also a configuration file so that few, if any command-line arguments are needed.
This also contains a `run_name` variable which is used to name the experiment. This name is then used to create a directory to store model checkpoints and results.

If the software is repeated with the same `run_name` then the software might not work, or might over write results.

## Extending the project.
The idea of the Git project with this format is to be able to easily trial new approaches by swapping/modifying the individual components.

For example, it might be that we want to change the approach to use the head of the data rather than the tail but we want to keep both separate.

To do so, we use git branches. 

When we want to try out a new idea we create a new branch, modify the codebase and push to the remote branch. On the server we then checkout the branch and run the code.