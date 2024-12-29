# MNIST-dataset-fully-connected-neural-network
This project implements a fully connected neural network in PyTorch to recognize handwritten digits from the MNIST dataset. It includes steps for data loading, model creation, training, and evaluation.

## Features
- Uses PyTorch for model implementation.
- Trains a simple FCNN with one hidden layer.
- Evaluates model accuracy on both training and test datasets.

## Prerequisites
- Python 3.7+
- PyTorch
- torchvision

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/banualy/MNIST-dataset-fullyconnectedneuralnetwork
   cd MNIST-dataset-fullyconnectedneuralnetwork
   ```

2. Install the required dependencies:
   ```bash
   pip install torch torchvision
   ```

## Usage
### Training the Model
1. Run the script to train the FCNN on the MNIST dataset:
   ```bash
   python test.py
   ```

2. The model will train for 3 epochs (default) and print training and test accuracies.

### Modifying Hyperparameters
You can adjust the following hyperparameters directly in the script:
- **input_size:** Size of the input features (default: `784`).
- **num_classes:** Number of output classes (default: `10`).
- **learning_rate:** Learning rate for the optimizer (default: `0.001`).
- **batch_size:** Number of samples per batch (default: `64`).
- **num_epochs:** Number of epochs for training (default: `3`).

### Evaluation
After training, the script evaluates the model's accuracy on both the training and test datasets.

## Code Structure
- **Model:** A simple FCNN with one hidden layer of 50 neurons.
- **Data Loader:** Loads and preprocesses the MNIST dataset using `torchvision.datasets`.
- **Training Loop:** Performs forward pass, computes loss, backpropagates, and updates weights.
- **Accuracy Check:** Evaluates model performance on training and test data.

## Results
After training, the model achieves approximately **98% accuracy** on the training data and **97% accuracy** on the test data (values may vary slightly).

## Contributing
Feel free to submit issues or pull requests to improve this project. Contributions are always welcome!




