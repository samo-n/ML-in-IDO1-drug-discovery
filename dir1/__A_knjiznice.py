### 1. Uvoz knjiznic in definiranje osnovnih funkcij

#RDKit
import rdkit
from rdkit import Chem
from rdkit.Chem import DataStructs, RDConfig, rdBase, Draw
from rdkit.Chem.Draw import IPythonConsole, MolDrawing, DrawingOptions
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from rdkit.Chem import Draw

#Sklearn
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import numpy as np
from numpy import asarray
import os
from itertools import cycle
from sklearn.preprocessing import label_binarize
import time
import csv

#Chembl
import math
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from chembl_webresource_client.new_client import new_client
from tqdm.auto import tqdm

#Metrics
from matplotlib import pyplot
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.metrics import balanced_accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from itertools import cycle
import gc

import xml.etree.ElementTree as ET
import pandas as pd


#Outlier removal
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

#Modeling
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn import model_selection, metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
# from scipy import interp
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier


#Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

#Keras

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# for reading data


from sklearn.utils import shuffle

# multi-class classification with Keras
import pandas


#Sampling techniques
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import BorderlineSMOTE


#Pandas, numpy
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from matplotlib import pyplot as plt

import os
import sys

#Analiza podatkov
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest

import tensorflow as tf
from tensorflow import keras
# for reading data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# for modeling
import pickle

#SMOTE
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import StandardScaler

import pickle
from sklearn.inspection import permutation_importance
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import RFE

import multiprocessing


# miscellaneous 
import math
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import warnings
from rdkit import Chem
from rdkit.Chem import PandasTools

from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

import math
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, PandasTools

from pathlib import Path
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import optuna