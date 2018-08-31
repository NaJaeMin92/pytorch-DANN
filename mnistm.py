import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import torch.utils.data as data
import torch
import os
import errno
from PIL import Image
import params


# MNIST-M
class MNISTM(data.Dataset):
    """`MNIST-M Dataset."""

    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'mnist_m_train.pt'
    test_file = 'mnist_m_test.pt'

    def __init__(self,
                 root, mnist_root="data",
                 train=True,
                 transform=None, target_transform=None,
                 download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__()
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.expanduser(mnist_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = \
                torch.load(os.path.join(self.root,
                                        self.processed_folder,
                                        self.training_file))
        else:
            self.test_data, self.test_labels = \
                torch.load(os.path.join(self.root,
                                        self.processed_folder,
                                        self.test_file))

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(type(img))
        img = Image.fromarray(img.squeeze().numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,
                                           self.processed_folder,
                                           self.training_file)) and \
               os.path.exists(os.path.join(self.root,
                                           self.processed_folder,
                                           self.test_file))

    def download(self):
        """Download the MNIST data."""
        # import essential packages
        from six.moves import urllib
        import gzip
        import pickle
        from torchvision import datasets

        # check if dataset already exists
        if self._check_exists():
            return

        # make data dirs
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # download pkl files
        print('Downloading ' + self.url)
        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(file_path.replace('.gz', '')):
            data = urllib.request.urlopen(self.url)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        # load MNIST-M images from pkl file
        with open(file_path.replace('.gz', ''), "rb") as f:
            mnist_m_data = pickle.load(f, encoding='bytes')
        mnist_m_train_data = torch.ByteTensor(mnist_m_data[b'train'])
        mnist_m_test_data = torch.ByteTensor(mnist_m_data[b'test'])

        # get MNIST labels
        mnist_train_labels = datasets.MNIST(root=self.mnist_root,
                                            train=True,
                                            download=True).train_labels
        mnist_test_labels = datasets.MNIST(root=self.mnist_root,
                                           train=False,
                                           download=True).test_labels

        # save MNIST-M dataset
        training_set = (mnist_m_train_data, mnist_train_labels)
        test_set = (mnist_m_test_data, mnist_test_labels)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('MNISTM Done!')


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.29730626, 0.29918741, 0.27534935),
                                                     (0.32780124, 0.32292358, 0.32056796))
                                ])

mnistm_train_dataset = MNISTM(root='data/pytorch/MNIST-M', train=True, download=True,
                              transform=transform)
mnistm_valid_dataset = MNISTM(root='data/pytorch/MNIST-M', train=True, download=True,
                              transform=transform)
mnistm_test_dataset = MNISTM(root='data/pytorch/MNIST-M', train=False, transform=transform)

indices = list(range(len(mnistm_train_dataset)))
validation_size = 5000
train_idx, valid_idx = indices[validation_size:], indices[:validation_size]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

mnistm_train_loader = DataLoader(
    mnistm_train_dataset,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

mnistm_valid_loader = DataLoader(
    mnistm_valid_dataset,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

mnistm_test_loader = DataLoader(
    mnistm_test_dataset,
    batch_size=params.batch_size,
    num_workers=params.num_workers
)

# print(mnistm_train_dataset.train_data[5000:].shape)
# mnistm_concat = (mnistm_train_dataset.train_data[5000:])


# def test():
#     print(mnistm_train_dataset.train_data[5000:].shape)
#     print((mnistm_train_dataset.train_data[5000:].size()))
#
#     print(len(train_sampler), len(mnistm_test_loader), len(valid_sampler))
#     print(len(mnistm_train_loader), len(mnistm_valid_loader), len(mnistm_test_loader))
#     for i in range(1):
#         for batch_idx, (inputs, labels) in enumerate(mnistm_train_loader):
#             print(i, batch_idx, labels, len(labels))
#     for batch_idx, (train_data, test_data) in enumerate(zip(mnistm_train_loader, mnistm_valid_loader)):
#         train_image, train_label = train_data
#         test_image, test_label = test_data
#         print(train_label, len(train_label))
#         print(test_label, len(test_label))
#         exit()

# test()
