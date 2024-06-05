import os, sys
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import cufflinks as cf
# cf.go_offline()
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
# Tính toán FPR và TPR từ decision function
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam




import tensorflow as tf

# from pydantic_settings import BaseSettings # NEW

# from pandas_profiling import ProfileReport

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder,StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout,Concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import RFE

from tensorflow import keras
from keras.activations import relu
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import layers as Layers
from numpy import set_printoptions
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone
# from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import StackingClassifier

# from kerastuner.tuners import RandomSearch
from scipy.stats import uniform
import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Flatten
from keras.models import Model
from keras.layers import Input
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold


from tabnet_keras import TabNetRegressor, TabNetClassifier
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone
# from mlxtend.classifier import StackingClassifier
# from kerastuner.tuners import RandomSearch
from scipy.stats import uniform
import numpy as np
import wandb
import enum
import math
import time
from copy import deepcopy
import warnings
import typing as ty
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim
# from torch import Tensor
# from .fttransformer import Transformer
# from .resnet import ResNet

# ModuleType = Union[str, Callable[..., nn.Module]]

# def reglu(x : Tensor) -> Tensor:
#     assert x.shape[-1] % 2 == 0
#     a, b = x.chunk(2, dim = -1)
#     return a * F.relu(b)

# def geglu(x : Tensor) -> Tensor:
#     assert x.shape[-1] % 2 == 0
#     a, b = x.chunk(2, dim = -1)
#     return a * F.gelu(b)

# class ReGLU(nn.Module):
#     def forward(self, x : Tensor) -> Tensor:
#         return reglu(x)
        
# class GEGLU(nn.Module):
#     def forward(self, x : Tensor) -> Tensor:
#         return geglu(x)

# def load_model(config: ty.Dict , info_dict : ty.Dict):
#     # if config["model"] == "ft-transformer":
#         # return Transformer(
#         #                 d_numerical = int(info_dict["n_num_features"]),
#         #                 categories = None,

#         #                 # Model Architecture
#         #                 n_layers = int(config["n_layers"]),
#         #                 n_heads = int(config["n_heads"]),
#         #                 d_token = int(config["d_token"]),
#         #                 d_ffn_factor = float(config["d_ffn_factor"]),
#         #                 attention_dropout = float(config["attention_dropout"]),
#         #                 ffn_dropout = float(config["attention_dropout"]),
#         #                 residual_dropout = float(config["residual_dropout"]),
#         #                 activation = config["activation"],
#         #                 prenormalization = True,
#         #                 initialization = config["initialization"],
                        
#         #                 # default_Setting
#         #                 token_bias = True,
#         #                 kv_compression = None if int(config["kv_compression"]) == 0 else int(config["kv_compression"]),
#         #                 kv_compression_sharing= None if int(config["kv_compression"]) == 0 else float(config["kv_compression"]),
#         #                 d_out = int(info_dict["n_classes"]) if info_dict["task_type"] == "multiclass" else 1
#         # )

#     if config["model"] == "resnet":
#         return ResNet(
#                     d_numerical= int(info_dict["n_num_features"]),
#                     categories = None,

#                     # ModelA Architecture
#                     activation = "relu",
#                     d = int(config["d"]),
#                     d_embedding = int(config["d_embedding"]),
#                     d_hidden_factor = float(config["d_hidden_factor"]), 
#                     hidden_dropout = float(config["hidden_dropout"]),
#                     n_layers = int(config["n_layers"]),
#                     normalization = config["normalization"],
#                     residual_dropout = float(config["residual_dropout"]),

#                     # default_Setting
#                     d_out = int(info_dict["n_classes"]) if info_dict["task_type"] == "multiclass" else 1
#         )
#     else:
#         pass




# import enum
# import math
# import time
# import warnings
# from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim
# from torch import Tensor
# # from libs.common import *

# def reglu(x: Tensor) -> Tensor:
#     a, b = x.chunk(2, dim=-1)
#     return a * F.relu(b)


# def geglu(x: Tensor) -> Tensor:
#     a, b = x.chunk(2, dim=-1)
#     return a * F.gelu(b)


# class ReGLU(nn.Module):
#     def forward(self, x: Tensor) -> Tensor:
#         return reglu(x)


# class GEGLU(nn.Module):
#     def forward(self, x: Tensor) -> Tensor:
#         return geglu(x)


# def get_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
#     return (
#         reglu
#         if name == 'reglu'
#         else geglu
#         if name == 'geglu'
#         else torch.sigmoid
#         if name == 'sigmoid'
#         else getattr(F, name)
#     )


# def get_nonglu_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
#     return (
#         F.relu
#         if name == 'reglu'
#         else F.gelu
#         if name == 'geglu'
#         else get_activation_fn(name)
#     )


# def get_nonglu_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
#     return (
#         F.relu
#         if name == 'reglu'
#         else F.gelu
#         if name == 'geglu'
#         else get_activation_fn(name)
#     )

# class ResNet(nn.Module):
#     def __init__(
#         self,
#         *,
#         d_numerical: int,
#         categories: ty.Optional[ty.List[int]],
#         d_embedding: int,
#         d: int,
#         d_hidden_factor: float,
#         n_layers: int,
#         activation: str,
#         normalization: str,
#         hidden_dropout: float,
#         residual_dropout: float,
#         d_out: int,
#     ) -> None:
#         super().__init__()

#         def make_normalization():
#             return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
#                 normalization
#             ](d)

#         self.main_activation = get_activation_fn(activation)
#         self.last_activation = get_nonglu_activation_fn(activation)
#         self.residual_dropout = residual_dropout
#         self.hidden_dropout = hidden_dropout

#         d_in = d_numerical
#         d_hidden = int(d * d_hidden_factor)

#         if categories is not None:
#             d_in += len(categories) * d_embedding
#             category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
#             self.register_buffer('category_offsets', category_offsets)
#             self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
#             nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
#             print(f'{self.category_embeddings.weight.shape=}')

#         self.first_layer = nn.Linear(d_in, d)
#         self.layers = nn.ModuleList(
#             [
#                 nn.ModuleDict(
#                     {
#                         'norm': make_normalization(),
#                         'linear0': nn.Linear(
#                             d, d_hidden * (2 if activation.endswith('glu') else 1)
#                         ),
#                         'linear1': nn.Linear(d_hidden, d),
#                     }
#                 )
#                 for _ in range(n_layers)
#             ]
#         )
#         self.last_normalization = make_normalization()
#         self.head = nn.Linear(d, d_out)

#     def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
#         x = []
#         if x_num is not None:
#             x.append(x_num)
#         if x_cat is not None:
#             x.append(
#                 self.category_embeddings(x_cat + self.category_offsets[None]).view(
#                     x_cat.size(0), -1
#                 )
#             )
#         x = torch.cat(x, dim=-1)

#         x = self.first_layer(x)
#         for layer in self.layers:
#             layer = ty.cast(ty.Dict[str, nn.Module], layer)
#             z = x
#             z = layer['norm'](z)
#             z = layer['linear0'](z)
#             z = self.main_activation(z)
#             if self.hidden_dropout:
#                 z = F.dropout(z, self.hidden_dropout, self.training)
#             z = layer['linear1'](z)
#             if self.residual_dropout:
#                 z = F.dropout(z, self.residual_dropout, self.training)
#             x = x + z
#         x = self.last_normalization(x)
#         x = self.last_activation(x)
#         x = self.head(x)
#         return x
    


