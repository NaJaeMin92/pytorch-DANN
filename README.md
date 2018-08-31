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
|Method     | Target Acc(paper) | Target Acc(this work)|
|:----------:|:-----------------:|:---------------------:|
|Source Only| 0.5225            | 0.512|
|DANN       | 0.7666            | 0.798|``````

### Experiments
`Source only`
![source_only](https://user-images.githubusercontent.com/37066691/44932869-febded80-ada1-11e8-8297-050b4a5ed8f7.png)

`DANN`
![dann](https://user-images.githubusercontent.com/37066691/44932883-08dfec00-ada2-11e8-8f32-fd6afe044224.png)

