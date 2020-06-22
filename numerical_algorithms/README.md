# Numerical Algorithms

I implemented a small neural network which uses Extended Kalman Filter (EKF) in order to tune the weights of the network. The neural network is used in turn to approximate a signal. The main finding of this project was that you can train a good neural network based signal estimator with a smaller network and fewer training iterations with this approach in comparison to the gradient descent based neural networks. However, the computational costs associated with EKF makes it impractical to use in a large neural network architectures.

- **Keywords**: Kalman Filter, Neural Networks, Signal Estimation
- **Technologies**: Python, PyTorch, Numpy

## Course Details

- **Module Name**: Numerical Algorithms and Methods to Identify Dynamical Systems for Data Analysis and Reconciliation
- **Module Instructor**: Prof. Dr.-Ing. Paolo Mercorelli
- **Project Year**: 2019/20

## File/Folder Details

- The project report will be uploaded once it has been assessed by the course instructor.
- Folder `ekfNN` contains the helper functions used in the latter three notebooks mentioned below.
- File `Plots of Signals.ipynb` contains the code used to generate the signals used to train and test the models.
- File `ekfNN_First Order Signal.ipynb` contains the code used to train and evaluate the models for First Order Signal.
- File `ekfNN_Second Order Signal.ipynb` contains the code used to train and evaluate the models for Second Order Signal.
- File `ekfNN_Sinusoidal Signal.ipynb` contains the code used to train and evaluate the models for Sinusoidal Signal.
