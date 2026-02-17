import torch.nn as nn

def _c1_int(c1):
    if isinstance(c1, (list, tuple)):
        c1 = c1[-1]
    return int(c1)

class ResBlock(nn.Module):
    def __init__(self, c1):
        super().__init__()
        c1 = _c1_int(c1)
        self.cv1 = nn.Conv2d(c1, c1, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act = nn.ReLU(inplace=True)
        self.cv2 = nn.Conv2d(c1, c1, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c1)

    def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        y = self.bn2(self.cv2(y))
        return self.act(x + y)

class ConvNeXtBlockLite(nn.Module):
    def __init__(self, c1):
        super().__init__()
        c1 = _c1_int(c1)
        self.dw = nn.Conv2d(c1, c1, 7, 1, 3, groups=c1)
        self.bn = nn.BatchNorm2d(c1)
        self.pw1 = nn.Conv2d(c1, 4 * c1, 1, 1, 0)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * c1, c1, 1, 1, 0)

    def forward(self, x):
        y = self.dw(x)
        y = self.bn(y)
        y = self.pw2(self.act(self.pw1(y)))
        return x + y

class MBV3BlockLite(nn.Module):
    def __init__(self, c1, exp=4):
        super().__init__()
        c1 = _c1_int(c1)
        mid = c1 * exp
        self.pw1 = nn.Sequential(
            nn.Conv2d(c1, mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid),
            nn.Hardswish(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(mid, mid, 3, 1, 1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.Hardswish(inplace=True),
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(mid, c1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        y = self.pw2(self.dw(self.pw1(x)))
        return self.act(x + y)
