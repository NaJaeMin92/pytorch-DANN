# DANN

PyTorch implementation of DANN (Domain-Adversarial Training of Neural Networks)

"[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf)"

"[Domain-Adversarial Training of Neural Networks](http://jmlr.org/papers/volume17/15-239/15-239.pdf)"



### Prerequisites

```
python 3.5
pytorch 0.4.1
```

### Results
`MNIST -> MNIST-M`

![thiswork](https://user-images.githubusercontent.com/37066691/44933106-db477280-ada2-11e8-8022-6306579a1919.png)

Result of this work is from avg(five tests below).

### Experiments
Source only
![source_only](https://user-images.githubusercontent.com/37066691/44932869-febded80-ada1-11e8-8297-050b4a5ed8f7.png)

DANN
![dann](https://user-images.githubusercontent.com/37066691/44932883-08dfec00-ada2-11e8-8f32-fd6afe044224.png)

1. MNIST and MNIST-M feature distributions
2. Source only
3. Domain adaptation (DANN)
![res](https://user-images.githubusercontent.com/37066691/44933409-ef3fa400-ada3-11e8-8317-ce4c71fbcc6c.png)
