# DANN-PyTorch :fire:
PyTorch implementation of DANN (Domain-Adversarial Training of Neural Networks)

> **[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf)**</br>
> Yaroslav Ganin, Victor Lempitsky</br>
> *In PMLR-2015*

> **[Domain-Adversarial Training of Neural Networks](http://jmlr.org/papers/volume17/15-239/15-239.pdf)**</br>
> Yaroslav Ganin et al.</br>
> *In JMLR-2016*


## Getting started

### Installation
Install library versions that are compatible with your environment.
```bash
git clone https://github.com/NaJaeMin92/pytorch-DANN.git
cd pytorch-DANN
conda create -n dann python=3.7
conda activate dann
pip install -r requirements.txt

```

### Recommended configuration

```
python=3.7
pytorch=1.12.1
matplotlib=3.2.2
sklearn=1.0.2
```

### Usages
Running the code below will execute both `source-only` and `DANN` training and testing:
```
python main.py
# You can adjust training settings in 'params.py', including batch size and the number of training epochs.
```

### t-SNE (t-distributed Stochastic Neighbor Embedding)
Our code includes the functionality to visualize `t-SNE`, both before and after the process of domain adaptation using `sklearn.manifold`.

## Experimental results
`MNIST -> MNIST-M`

| Method                      | Test #1   | Test #2   | Test #3   | Test #4   | Test #5     | Avg.        |
| :-------------------------: | :-------: | :-------: | :-------: | :-------: | :---------: | :---------: | 
| Source Accuracy             | 89        | 98        | 98        | 90        | 98          | **61.2**    | 
| Target Accuracy             | 47        | 56        | 54        | 46        | 53          | **51.2**    | 

DANN
| Method                      | Test #1   | Test #2   | Test #3   | Test #4   | Test #5     | Avg.        |
| :-------------------------: | :-------: | :-------: | :-------: | :-------: | :---------: | :---------: | 
| Source Accuracy             | 96        | 96        | 97        | 97        | 96          | **96.4**    | 
| Target Accuracy             | 83        | 78        | 80        | 80        | 78          | **79.8**    | 
| Domain Accuracy             | 60        | 60        | 61        | 64        | 61          | **61.2**    | 
