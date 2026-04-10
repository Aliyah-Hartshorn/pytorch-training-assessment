# pytorch-training-assessment



## Part A Code Explanation

This Python script is a complete deep learning training pipeline built using PyTorch, designed to train a Convolutional Neural Network (CNN) on the CIFAR-10 image classification dataset. Below is a breakdown of each section: 



##### Imports 

The script begins by importing a range a libraries, each serving a specific purpose: 



* argparse- allows the script to accepts command like arguments (like number of epochs or learning rate) when run from a terminal. 
* os – handles file system operations such as creating directories and building file paths.
* random – used for generating random numbers, particularly for reproducibility.
* json – used to save training metrics in a structured JSON file.
* datetime – used to timestamp each training epoch in the console output.
* numpy (np) – provides support for numerical operations and array handling.
* torch / torch.nn / torch.optim – the core PyTorch libraries used to define, train, and optimise the neural network.
* DataLoader / datasets / transforms – PyTorch utilities for loading, augmenting, and batching the dataset.
* confusion\_matrix (sklearn) – generates a confusion matrix to evaluate model performance across classes.
* matplotlib.pyplot – used to visualise and save the confusion matrix as an image.

