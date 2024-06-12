import math
import sys
from typing import Iterable
import torch
import util.misc as misc
import util.lr_sched as lr_sched
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from math import ceil
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageNet,ImageFolder
from torch.utils.data import Dataset, DataLoader, Subset
import timm
import random
import timm.optim.optim_factory as optim_factory
import models_vit
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed
import masknet_mae as models_mae
import util.lr_decay as lrd
# from engine_pretrain import train_one_epoch
# from engine_finetune import train_one_epoch as train_one_epoch_linprobe, evaluate
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from collections import defaultdict
from typing import Iterable, Optional
from timm.data import Mixup
import torch.nn.functional as F
from ema import ExponentialMovingAverage
