from enum import Enum
from typing import TypedDict

from torch import Tensor


class DatasetStage(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Views(TypedDict, total=False):
    extrinsics: Tensor
    intrinsics: Tensor
    image: Tensor
    near: Tensor
    far: Tensor
    index: Tensor


class Example(TypedDict, total=False):
    context: Views
    target: Views
    scene: str
    all_ind: int
