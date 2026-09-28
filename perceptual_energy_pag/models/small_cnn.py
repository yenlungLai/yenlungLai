import torch.nn as nn

# Fixed dataset statistics of the training split (see config.TrainConfig). A constant affine map: it adds no
# invariance, unlike per-image standardisation which would make the classifier exactly contrast/offset invariant.
INPUT_MEAN, INPUT_STD = 0.345, 0.107


def _block(i, o):
    return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]


class SmallCNN(nn.Module):
    """Identical architecture for CE / AT / PAG. ~60k parameters, receptive field ~22 px before GAP.
    BatchNorm is used for optimisation only; all analyses run in eval mode where BN is a fixed affine map."""

    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            *_block(1, 16), *_block(16, 32), nn.MaxPool2d(2),
            *_block(32, 64), nn.MaxPool2d(2),
            *_block(64, 64),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(self.features((x - INPUT_MEAN) / INPUT_STD).mean(dim=(2, 3)))
