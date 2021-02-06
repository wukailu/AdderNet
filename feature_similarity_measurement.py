import numpy as np
import torch
from torch import Tensor


# Return range [0, 1] [similar, not similar], cka_loss(a, a) = 0
class cka_loss(torch.nn.Module):
    def __init__(self, rbf_threshold=-1):
        super().__init__()
        self.rbf_threshold = rbf_threshold

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        y = y.view(batch_size, -1)
        if self.rbf_threshold < 0:
            return self._cka(self._gram_linear(x), self._gram_linear(y))
        else:
            return self._cka(self._gram_rbf(x, self.rbf_threshold), self._gram_rbf(y, self.rbf_threshold))
        pass

    @staticmethod
    def _gram_linear(features_x: Tensor):
        return features_x.mm(features_x.transpose(0, 1))

    @staticmethod
    def _center_gram(gram: Tensor):
        n = gram.size(0)
        gram.fill_diagonal_(0)
        means = torch.sum(gram, dim=0) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        gram.fill_diagonal_(0)
        return gram

    @staticmethod
    def _gram_rbf(feature_x: Tensor, threshold=1):
        dot_products = feature_x.mm(feature_x.transpose(0, 1))
        sq_norms = torch.diag(dot_products)
        sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
        sq_median_distance = torch.median(sq_distances)
        return torch.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))

    def _cka(self, gram_x: Tensor, gram_y: Tensor):
        gram_x = gram_x.float()
        gram_y = gram_y.float()
        gram_x = self._center_gram(gram_x)
        gram_y = self._center_gram(gram_y)

        scaled_hsic = gram_x.view(-1).dot(gram_y.view(-1))

        normalization_x = torch.norm(gram_x) + 1e-5
        normalization_y = torch.norm(gram_y) + 1e-5

        assert 0 < normalization_x
        assert 0 < normalization_y
        # decreasing 0 -> similar 1 -> not similar
        return 1 - scaled_hsic / (normalization_x * normalization_y)


class layer_loss(torch.nn.Module):
    def __init__(self, layer, criterion=torch.nn.MSELoss()):
        super().__init__()
        self.layer = layer
        self.criterion = criterion

    def forward(self, f_student: Tensor, f_teacher: Tensor) -> Tensor:
        return self.criterion(self.layer(f_student), f_teacher)
