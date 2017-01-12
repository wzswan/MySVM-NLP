"""
MYSVM: An implementation of multiple-instance support vector machines

The following algorithms are implemented:

  SVM     : a standard supervised SVM
  SIL     : trains a standard SVM classifier after applying bag labels to each
            instance

"""
__name__ = 'mysvm'
__version__ = '1.0'
from svm import SVM
from sil import SIL

