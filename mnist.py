import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import torch
import params

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.29730626, 0.29918741, 0.27534935),
                                                     (0.32780124, 0.32292358, 0.32056796)),
                                ])

mnist_train_dataset = datasets.MNIST(root='data/pytorch/MNIST', train=True, download=True,
                                     transform=transform)
mnist_valid_dataset = datasets.MNIST(root='data/pytorch/MNIST', train=True, download=True,
                                     transform=transforms)
mnist_test_dataset = datasets.MNIST(root='data/pytorch/MNIST', train=False, transform=transform)

indices = list(range(len(mnist_train_dataset)))
validation_size = 5000
train_idx, valid_idx = indices[validation_size:], indices[:validation_size]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

mnist_train_loader = DataLoader(
    mnist_train_dataset,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

mnist_valid_loader = DataLoader(
    mnist_valid_dataset,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

mnist_test_loader = DataLoader(
    mnist_test_dataset,
    batch_size=params.batch_size,
    num_workers=params.num_workers
)


# mnist_train_all = (mnist_train_dataset.train_data[5000:].reshape(55000, 28, 28, 1))
# mnist_concat = torch.cat((mnist_train_all, mnist_train_all, mnist_train_all), 3)
# print(mnist_test_dataset.test_labels.shape, mnist_test_dataset.test_labels)


def one_hot_embedding(labels, num_classes=10):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


# print(one_hot_embedding(mnist_test_dataset.test_labels))

# print(mnist_concat.shape)


# def test():
    # print(mnist_train_loader.shape)
    # print(len(train_sampler), len(mnist_test_loader), len(valid_sampler))
    # print(len(mnist_train_loader), len(mnist_valid_loader), len(mnist_test_loader))
    # for i, train_data in enumerate(mnist_train_loader):
    #     img, label = train_data
    #     print(img.shape)
    # for i in range(1):
    #     # for batch_idx, (inputs, labels) in enumerate(train_loader):
    #     #     print(i, batch_idx, labels, len(labels))
    # mnist_train_all = (mnist_train_dataset.train_data[5000:].reshape(55000, 28, 28, 1))
    # mnist_concat = torch.cat((mnist_train_all, mnist_train_all, mnist_train_all), 3)
    # print(mnist_concat.shape)
    # print(list(mnist_train_dataset.train_data[5000:].size()))
    # print(mnist_train_dataset.train_data.float().mean()/255)
    # print(mnist_train_dataset.train_data.float().std()/255)
    # for batch_idx, (train_data, test_data) in enumerate(zip(mnist_train_loader, mnist_valid_loader)):
    #     train_image, train_label = train_data
    #     test_image, test_label = test_data
    #     print(train_image.shape)
    #     # print(train_label, len(train_label))
    #     # print(test_label, len(test_label))
    #     # exit()

# test()
