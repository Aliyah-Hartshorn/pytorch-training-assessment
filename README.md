Part A Assessment – Detailed Code Explanation

This Python script is a complete deep learning training pipeline built using PyTorch, designed to train a Convolutional Neural Network (CNN) on the CIFAR-10 image classification dataset. Below is a thorough breakdown of each section.



1\. Imports

The script begins by importing a range of libraries, each serving a specific purpose:



argparse – allows the script to accept command-line arguments (like number of epochs or learning rate) when run from a terminal.

os – handles file system operations such as creating directories and building file paths.

random – used for generating random numbers, particularly for reproducibility.

json – used to save training metrics in a structured JSON file.

datetime – used to timestamp each training epoch in the console output.

numpy (np) – provides support for numerical operations and array handling.

torch / torch.nn / torch.optim – the core PyTorch libraries used to define, train, and optimise the neural network.

DataLoader / datasets / transforms – PyTorch utilities for loading, augmenting, and batching the dataset.

confusion\_matrix (sklearn) – generates a confusion matrix to evaluate model performance across classes.

matplotlib.pyplot – used to visualise and save the confusion matrix as an image.





2\. Reproducibility – set\_seed()

pythondef set\_seed(seed=42):

This function ensures that every time the script is run, it produces the same results. It does this by fixing the random seed across all libraries — Python's built-in random, NumPy, and PyTorch (both CPU and GPU). The deterministic = True setting forces PyTorch to use consistent algorithms, while benchmark = False disables the auto-tuner that could introduce variability.



3\. Model Definition – SimpleCNN

pythonclass SimpleCNN(nn.Module):

This defines the neural network architecture as a class inheriting from PyTorch's nn.Module. It is split into two parts:

Feature Extractor (self.features):



Two convolutional layers (Conv2d) extract spatial features from the images, such as edges and textures.

Each convolutional layer is followed by a ReLU activation function, which introduces non-linearity, helping the model learn complex patterns.

MaxPool2d layers reduce the spatial dimensions by half after each convolution, keeping only the most prominent features and reducing computation.



Classifier (self.classifier):



Flatten converts the 3D feature maps into a 1D vector so they can be passed into fully connected layers.

Two Linear (fully connected) layers process the flattened features, with a ReLU activation in between.

The final linear layer outputs 10 values — one for each class in CIFAR-10.



Forward Pass (forward):



This method defines how data flows through the network, passing input through the feature extractor and then the classifier.





4\. Training Loop – train\_one\_epoch()

pythondef train\_one\_epoch(model, loader, criterion, optimizer, device):

This function handles one full pass through the training data:



model.train() sets the model to training mode, enabling features like dropout if present.

For each batch, images and labels are sent to the appropriate device (CPU or GPU).

optimizer.zero\_grad() clears old gradients before each update to prevent accumulation.

The model makes predictions (outputs), and the loss is calculated using the criterion.

loss.backward() computes gradients through backpropagation.

optimizer.step() updates the model's weights based on those gradients.

The function returns the average loss and accuracy for the epoch.





5\. Evaluation Loop – evaluate()

pythondef evaluate(model, loader, criterion, device):

This function assesses the model's performance on the test dataset:



model.eval() switches the model to evaluation mode, disabling training-specific behaviour.

torch.no\_grad() disables gradient calculations, saving memory and speeding up inference.

It collects all predictions and true labels across the test set, returning them alongside the average loss and accuracy. These are later used to generate the confusion matrix.





6\. Confusion Matrix – save\_confusion\_matrix()

pythondef save\_confusion\_matrix(labels, preds, classes, output\_dir):

This function creates a visual confusion matrix using Matplotlib:



confusion\_matrix() from scikit-learn compares the true labels against the model's predictions, producing a grid that shows where the model is getting things right and where it is confusing one class for another.

The matrix is displayed as a heatmap using imshow with a blue colour scheme, labelled with the class names on both axes.

The final image is saved as confusion\_matrix.png in the output directory.





7\. Main Function – main()

This is the core function that ties everything together and runs in the following sequence:

Argument Validation:

Checks that epochs, batch size, and learning rate are all positive values, raising a ValueError if not.

Setup:



Sets the random seed for reproducibility.

Creates the output directory if it doesn't already exist.

Detects and assigns the appropriate device (GPU via CUDA if available, otherwise CPU).



Data Transforms:



Training data is augmented with a random horizontal flip to improve generalisation, then converted to tensors and normalised.

Test data is only converted and normalised, with no augmentation.



Dataset \& DataLoader:



The CIFAR-10 dataset is downloaded and loaded for both training and testing.

DataLoader wraps the datasets to handle shuffling, batching, and parallel loading with 2 worker threads.



Model Setup:



A SimpleCNN instance is created and moved to the target device.

The total number of trainable parameters is printed for reference.

Cross-entropy loss is used as the criterion, Adam as the optimiser, and a step learning rate scheduler halves the learning rate every 2 epochs.



Training Loop:



For each epoch, the model is trained and then evaluated on the test set.

Results are printed to the console with a timestamp.

Metrics are logged to a list, and the best-performing model (by validation accuracy) is saved as best\_model.pth.



Saving Logs:

All epoch metrics are saved to a metrics.json file in the output directory for later review.

Confusion Matrix:

The confusion matrix is generated using the predictions and labels from the final epoch and saved as an image.

Inference Demo:



The best saved model is reloaded from the checkpoint.

A single test image is passed through the model to demonstrate inference, and the predicted class index is printed to the console.





8\. Command-Line Interface (CLI)

pythonif \_\_name\_\_ == "\_\_main\_\_":

This block only runs when the script is executed directly (not imported). It uses argparse to define and accept the following configurable parameters from the terminal:

ArgumentDefaultDescription--epochs5Number of training epochs--batch-size64Number of images per batch--lr0.001Learning rate for the optimiser--data./dataDirectory to store the dataset--output-dir./outputsDirectory to save outputs--trackernoneOptional experiment tracker (wandb/mlflow)



Summary

Overall, this Part A script is a well-structured, end-to-end machine learning pipeline. It covers all the key stages of a typical deep learning project — data loading and augmentation, model definition, training, evaluation, metric logging, and inference — while also incorporating best practices such as reproducibility seeding, model checkpointing, and command-line configurability.

