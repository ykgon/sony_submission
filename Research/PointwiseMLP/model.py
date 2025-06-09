import torch.nn as nn

class PointwiseMLP(nn.Module):
    """
    単変量 → 単変量回帰の MLP:
      Linear(1→H)→BN→ReLU→Linear(H→H)→BN→ReLU
      →Linear(H→H)→BN→ReLU→Linear(H→1)
    ※元コードでは bn3, bn4 定義のみで forward では bn1/bn2 のみ適用されていたため、
      同様に bn1, bn2 のみを適用した構成です。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.relu(h)

        h = self.fc2(h)
        h = self.bn2(h)
        h = self.relu(h)

        h = self.fc3(h)
        # 元コードではここに bn3/bn4 は適用せず直接出力
        h = self.fc4(h)
        return h
