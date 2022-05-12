"""
Street-View House Numbers.
PyTorch Docs:
https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html
"""

from torch.utils.data import SubsetRandomSampler, DataLoader
import torchvision.datasets as datasets
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# Do this to ensure that the strings are valid.
splitStrings = ['train', 'test', 'extra']

svhn_train_dataset = datasets.SVHN(
    root = '/data/pytorch/SVHN',
    split = splitStrings[0], # train
    download = True,
    transform = transform
)

svhn_valid_dataset = datasets.SVHN(
    root = '/data/pytorch/SVHN',
    split = splitStrings[0], # train
    download = True,
    transform = transform
)

svhn_test_dataset = datasets.SVHN(
    root = '/data/pytorch/SVHN',
    split = splitStrings[1], # train
    download = True,
    transform = transform
)

indices = list(range(len(svhn_train_dataset)))
validation_size = 5000
train_idx, valid_idx = indices[validation_size:], indices[:validation_size]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

svhn_train_loader = DataLoader(
    svhn_train_dataset,
    batch_size = params.batch_size,
    sampler = train_sampler,
    num_workers = params.num_workers
)

svhn_train_loader = DataLoader(
    svhn_train_dataset,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

svhn_valid_loader = DataLoader(
    svhn_valid_dataset,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

svhn_test_loader = DataLoader(
    svhn_test_dataset,
    batch_size=params.batch_size,
    num_workers=params.num_workers
)
