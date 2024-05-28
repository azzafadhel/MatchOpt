from util import *

class SequentialEmbeddingHead(nn.Module):
    def __init__(self, n_dim, embedding_dim=32):
        super(SequentialEmbeddingHead, self).__init__()
        self.n_dim = n_dim
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(n_dim, 16 * embedding_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(16 * embedding_dim, 4 * embedding_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.LeakyReLU(0.3)
        )

    def forward(self, x):
        return self.net(x)

class RegressionNet(nn.Module):
    def __init__(self, n_dim, embedding_dim=32):
        super(RegressionNet, self).__init__()
        self.head = SequentialEmbeddingHead(n_dim, embedding_dim)
        self.clf = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.clf(self.head(x))

class RankNet(nn.Module):
    def __init__(self, n_dim, embedding_dim=32):
        super(RankNet, self).__init__()
        self.head = SequentialEmbeddingHead(n_dim, embedding_dim)
        self.clf = nn.Linear(embedding_dim, 1)

    def forward(self, x1, x2):
        z1, z2 = self.head(x1), self.head(x2)
        return torch.sigmoid(torch.mean(z1.unsqueeze(1) - z2, dim=-1))