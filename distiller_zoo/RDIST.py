import torch.nn as nn
import torch
import torch.nn.functional as F


class RDIST(nn.Module):
    def __init__(self, wa=10, beta=1.0, gamma=1.0, T=4.0):
        super(RDIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.T = T
        self.wa = wa

    def forward(self, logits_s, logits_t, f_s, f_t):
        y_s = (logits_s / self.T).softmax(dim=1)
        y_t = (logits_t / self.T).softmax(dim=1)
        inter_loss = self.T ** 2 * inter_class_relation(y_s, y_t)
        intra_loss = self.T ** 2 * intra_class_relation(y_s, y_t)

        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # RKD Angle loss
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            # L2范数，然后每个元素除以该范数的平方根
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        loss_a = F.smooth_l1_loss(s_angle, t_angle)
        # loss_b = F.mse_loss(s_angle, t_angle)
        # loss_a = 0.0421   loss_b = 0.0842

        kd_loss = self.beta * inter_loss + self.gamma * intra_loss + self.wa * loss_a
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
