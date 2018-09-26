#from __future__ import unicode_literals
import sys
import copy
import time
#----------------------------
# fix random seed for reproducibility
import numpy as np
import random as rn
import tensorflow as tf
np.random.seed(1)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
#---------------------------                                                                                                           
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from sklearn.ensemble import BaggingClassifier
from keras.wrappers.scikit_learn import KerasClassifier
# Create first network with Keras
from keras.layers import Dense, Reshape, Activation, Dropout, LSTM, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import RMSprop, Adamax, Adagrad, SGD
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_auc_score, roc_curve, auc
import csv
#import pandas
#from keras.regularizers import l1, activity_l1, l1l2
import matplotlib.pyplot as plt
#from keras.utils.visualize_util import plot
import uproot
import pickle
from sklearn.metrics import confusion_matrix
import itertools
from matplotlib import colors
import glob

#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.unicode'] = True

#np.seterr(divide='ignore', invalid='ignore')
#np.set_printoptions(threshold='nan')

def drawROC_fpr(**kwargs):
    """
    Input:{'model_name':[fpr, tpr, thresholds, val_info]}
    """
    fig,ax = plt.subplots( nrows=1, ncols=1 )
    legend_text=[]
    for key, value in kwargs.iteritems():
        auc_score = auc(value[0], value[1], reorder=True)
        print("AUC on test %s: %f" % (key, auc_score))
        ax.plot(value[0], value[1])
        legend_text.append("%s: %.2f, val: %.2f" % (key, auc_score, value[-1][1]))
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('bkg selection (false positive rate)')
    ax.set_ylabel('signal selection (true positive rate)')
    ax.set_title('ROC')
    ax.legend(legend_text, loc='best')
    plt.grid(True)
    fig.savefig("ROC_fpr.png")
    plt.close(fig)
    #plt.show()
    return

def build_dict(model_name):
    model_dict={}
    for iname in model_name:
        if (len(glob.glob("Experiments/"+iname+"/file_fpr")) != 1):
               raise StandardError('There should be ONE and ONLY ONE ROC for each model!!!')
        else:
            fpr=pickle.load(open(glob.glob("Experiments/"+iname+"/file_fpr")[0] ) ) 
            tpr=pickle.load(open(glob.glob("Experiments/"+iname+"/file_tpr")[0] ) )
            threshold=pickle.load(open(glob.glob("Experiments/"+iname+"/file_threshold")[0] ) )
            val_info=pickle.load(open(glob.glob("Experiments/"+iname+"/file_val")[0] ) ) 
            model_dict[iname]=[fpr, tpr, threshold, val_info]

    return model_dict

if __name__ == '__main__':
    model_name = ['dnn_separate_input', 'dnn']
    model_dict = build_dict(model_name)
    
    drawROC_fpr(**model_dict)
