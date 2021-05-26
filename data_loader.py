# 목표 : DNN - MNIST 분류(Pytorch Ignite 활용)
# data_loader.py : 

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# DataSet에서 데이터 읽고, 전처리(데이터 나누기, Scaling)
class MnistDataset(Dataset):     # Dataset 클래스를 상속받는다.

    # 데이터 읽어오기
    def __init__(self, data, labels, flatten=True):   # flatten=True : MNIST 28x28 -> 784
        self.data = data
        self.labels = labels
        self.flatten = flatten

        super().__init__()

    # 데이터의 크기 알기
    def __len__(self):
        return self.data.size(0)    # 샘플 수

    # 전처리 및 미니배치를 위한 샘플 반환
    def __getitem__(self, idx):
        x = self.data[idx]          # |x|=(28, 28)
        y = self.labels[idx]        # |y|=(1, )

        if self.flatten:            # flatten을 거치고나면 x는
            x = x.view(-1)          # |x|=(784, ) 로 변환된다.

        return x, y                 # |x|=(784, ), |y|=(1, )

# MNIST 데이터를 불러오는 함수
def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(

        # train 데이터를 가져온다.
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.      # 0~1사이로 스케일 변환
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y

# 불러온 MNIST 데이터를 config 기반으로 shuffle, train,valid split
def get_loaders(config):
    x, y = load_mnist(is_train=True, flatten=False)

    # 비율을 구한 후 train, valid split
    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))                 # 학습 데이터 index : 60000장

    train_x, valid_x = torch.index_select(              # features 
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)              # |train_x|=(48000,28,28)
    train_y, valid_y = torch.index_select(              # target
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    # -> |train_x|=(48000,28,28), |train_x|=(12000,28,28)

    # DataLoader 클래스로 train 데이터 가져오기
    train_loader = DataLoader(
        dataset=MnistDataset(train_x, train_y, flatten=True),
        batch_size=config.batch_size,                          # batch_size
        shuffle=True,                                          # train은 무조건 shuffle=True
    )

    # DataLoader 클래스로 valid 데이터 가져오기
    valid_loader = DataLoader(
        dataset=MnistDataset(valid_x, valid_y, flatten=True),  
        batch_size=config.batch_size,
        shuffle=True,                                          # valid는 shuffle 해도 되고 안해도 되고
    )

    # DataLoader 클래스로 test 데이터 가져오기
    test_x, test_y = load_mnist(is_train=False, flatten=False)
    test_loader = DataLoader(
        dataset=MnistDataset(test_x, test_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=False,                                         # test는 shuffle 보통 안 함
    )

    # 최종적으로 전처리 완료된 MNIST 데이터를 train, valid, test 데이터로 나누어서 반환한다.
    return train_loader, valid_loader, test_loader
