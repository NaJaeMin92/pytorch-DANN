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
![input_tsne_plot](https://user-images.githubusercontent.com/37066691/44933136-ffa34f00-ada2-11e8-8135-b78534c0d660.png)

2. Source only
![image](https://user-images.githubusercontent.com/37066691/44933190-36796500-ada3-11e8-8d3c-c4d3aa3dbe16.png)

3. Domain adaptation (DANN)
![image](https://user-images.githubusercontent.com/37066691/44933227-527d0680-ada3-11e8-9eb9-239b01513d3d.png)
