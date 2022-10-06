# Meta-Heuristic Deep Application: A Lecture on Nature-Inspired Optimization Applied to Deep Learning

Meta-Heuristic Deep Application provides a thorough lecture on nature-inspired optimization applied to deep learning architectures to be given at "5ª Escola Avançada em Big Data Analytics - ICMC/USP". Apart from the lecture itself, we also implemented various tasks to illustrate how one can use such tools in real-world applications.

---

## Guidelines

1. Every needed piece of information is elucidated in this **README**;
2. **Installation** is also straightforward and well-explained;
3. If there is a problem, please do not **hesitate** and call us.

---

## Installation

The installation process is straightforward as the dependencies are listed on the `requirements.txt` file. One can install them under the most preferred Python environment (raw, conda, virtualenv):

```bash
pip install -r requirements.txt
```

Additionally, the source files for the lecture are presented in LaTeX. Thus, one might need an additional compiler or even Overleaf to build the files into a PDF file.

---

## Getting Started

This section provides an overview of the text augmentation lecture, as well as three text augmentation applications.

### Lecture

The lecture is written in Portuguese in a slide-based format. The contents are available in the `slides` folder and compiled to PDF using a LaTeX compiler.

### Meta-Heuristic Optimization

Meta-heuristic optimization provides a straightforward optimization architecture, where nature-inspired heuristics are used to find the most relevant decision variables of a fitness function. Its main advantage relies on working out-of-the-box with non-convex functions and balancing its search between exploitation and exploration. Such procedure is implemented and available in the `applications/hyperparameter_optimization` folder and corresponds to the benchmarking function optimization task.

### Deep Learning (NNs, CNNs, and RNNs)

Deep learning architectures are widely employed in the most diverse tasks, such as image and text recognition and object detection. The implemented models are available at `applications/deep_learning` and provides pre-ready classes applied to supervised learning, e.g., classification.

### Finding Suitable Hyperparameters

An interesting approach to combine deep learning and meta-heuristic optimization is by exploring the ability to find the most suitable hyperparameters of a neural network, such as learning rate, number of hidden neurons, size of convolutions, among others. Hence, this lecture also offers pre-built implementations at `applications/hyperparameter_optimization` to accomplish such a task.

### Feature Selection

Finally, another exciting application of meta-heuristic optimization is to select the most convenient features of a dataset. Analogous to hyperparameter optimization, one can employ a classifier and optimize its validation accuracy over several subsets of features, aiming to find the most descriptive ones. This application is implemented and available in the `applications/feature_selection` folder.

---

## Support

It is inevitable to make mistakes or create some bugs. If there is any problem or concern, we will be available at this repository.

---
