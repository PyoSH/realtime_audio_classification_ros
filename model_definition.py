import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # RNN의 마지막 출력을 사용
        return out
class CustomLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(CustomLSTMModel, self).__init__()
        # 첫 번째 LSTM 층: 입력 차원에서 128개의 은닉 유닛으로
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)

        # 두 번째 LSTM 층: 첫 번째 층의 128개의 출력을 받아 64개의 은닉 유닛으로
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)

        # 완전 연결 층
        self.fc = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # 첫 번째 LSTM 층을 통과
        x, _ = self.lstm1(x)

        # 두 번째 LSTM 층을 통과
        x, _ = self.lstm2(x)

        # 시퀀스의 마지막 요소만을 사용하여 출력 계산
        x = x[:, -1, :]

        # 완전 연결 층을 통과
        x = self.fc(x)

        return x