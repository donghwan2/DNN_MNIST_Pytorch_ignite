# 목표 : DNN - MNIST 분류(Pytorch Ignite 활용)
# 1_model.py : 입력값 x를 받아서 예측값 y를 출력해주는 DNN 다중 분류 모델 클래스 존재
# 784차원 입력 -> f -> 10차원 출력

import torch
import torch.nn as nn

# 입력값 x를 받아서 예측값 y를 출력해주는 DNN 다중 분류 모델 클래스
class ImageClassifier(nn.Module):

    # Classifier 모델의 layers 구조 정의
    def __init__(self,                      # init 함수는 다음 인자를 받는다.
                 input_size,                # 입력 사이즈 = 784
                 output_size):              # 출력 사이즈 = 10
        self.input_size = input_size
        self.output_size = output_size

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 500),     # 입력 784 차원 -> 500
            nn.LeakyReLU(),                 # LeakyReLU
            nn.BatchNorm1d(500),            # batchNorm

            nn.Linear(500, 400),            # 500 -> 400
            nn.LeakyReLU(),
            nn.BatchNorm1d(400),

            nn.Linear(400, 300),            # 400 -> 300
            nn.LeakyReLU(),
            nn.BatchNorm1d(300),

            nn.Linear(300, 200),            # 300 -> 200
            nn.LeakyReLU(),
            nn.BatchNorm1d(200),

            nn.Linear(200, 100),            # 200 -> 100
            nn.LeakyReLU(),
            nn.BatchNorm1d(100),

            nn.Linear(100, 50),             # 100 -> 50
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),

            nn.Linear(50, output_size),     # 50 -> 10

            nn.Softmax(dim=-1),          # 다중 분류이므로 softmax
        )

    # forward 함수 : x라는 입력이 들어오면 layer에 집어넣는 함수 (init에서 정의한 모델 구조에 따라 실제 계산 수행)
    def forward(self, x):
        # |x| = (batch_size, input_size)

        y = self.layers(x)
        # |y| = (batch_size, output_size)

        # 최종적으로 예측값(y_hat) 리턴
        return y
