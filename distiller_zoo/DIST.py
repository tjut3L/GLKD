import torch.nn as nn


class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, T=4.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.T = T

    def forward(self, logits_s, logits_t):
        y_s = (logits_s / self.T).softmax(dim=1)
        y_t = (logits_t / self.T).softmax(dim=1)
        inter_loss = self.T ** 2 * inter_class_relation(y_s, y_t)
        intra_loss = self.T ** 2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))
