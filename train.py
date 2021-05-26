# 목표 : DNN MINST 분류(Pytorch ignite 활용)

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier       # model.py
from trainer import Trainer             # trainer.py
from data_loader import get_loaders     # utils.py


# 사용자의 입력을 configuration으로 받아오는 함수
def define_argparser():
    p = argparse.ArgumentParser()

    # 모델 파일 이름(required=True)
    # $python train.py --model_fn ./model.pth
    p.add_argument('--model_fn', required=True)     # model.pth 파일

    # device 설정 : gpu cuda가 있으면 gpu_id를 입력 받는다. ($python train.py --gpu_id id)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)  # -1은 cpu

    # train/valid 데이터를 나눌 ratio
    p.add_argument('--train_ratio', type=float, default=.8)

    # 미니배치 사이즈는 default가 256 (수정 : $python train.py --batch_size 512)
    p.add_argument('--batch_size', type=int, default=256)

    # epoch 수 (수정 : $python train.py --n_epochs 30)
    p.add_argument('--n_epochs', type=int, default=20)

    # epoch 종료마다 제공되는 정보의 정도(iteration 끝날 때마다 정보를 보고 싶으면 2)
    p.add_argument('--verbose', type=int, default=2)

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    # MNIST 데이터 로드(from trainer.py)
    train_loader, valid_loader, test_loader = get_loaders(config)

    # 샘플 갯수 확인
    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))
    print("Test:", len(test_loader.dataset))

    # 모델,옵티마이저,오차 함수
    model = ImageClassifier(28**2, 10).to(device)  # 모델 : 입력:28**2, 아웃풋:10, 
    optimizer = optim.Adam(model.parameters())     # 옵티마이저 : Adam
    crit = nn.CrossEntropyLoss()                   # 오차 함수 : 다중 분류이므로 CEL    (참고) nn.NLLLoss()

    # trainer.py 파일 -> ignite 활용
    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader)

if __name__ == '__main__':

    # main 함수가 시작될 때 위에서 정의한 config를 받아온다.
    config = define_argparser()

    # 인자로 config를 넣어주면 main 함수가 실행된다.
    main(config)
