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
import itertools
from matplotlib import colors



def drawROCCompare(Y_test, p_test, weight_test, ClassifBDTOutput_inclusive_withBTag_new_test):

    fpr1, tpr1, thresholds1 = roc_curve(Y_test[:, 0], p_test[:, 0], sample_weight=weight_test)
    fpr5, tpr5, thresholds5 = roc_curve(Y_test[:, 0], ClassifBDTOutput_inclusive_withBTag_new_test, sample_weight=weight_test)

    rnnScore = roc_auc_score(Y_test[:, 0], p_test[:, 0], sample_weight=weight_test)
    print("RNN AUC on test: %f" % rnnScore)
    bdtScore = roc_auc_score(Y_test[:, 0], ClassifBDTOutput_inclusive_withBTag_new_test, sample_weight=weight_test)
    print("BDT AUC on test: %f" % bdtScore)

    
    fig,ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(fpr1, tpr1)
    ax.plot(fpr5, tpr5)
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('bkg selection (false positive rate)')
    ax.set_ylabel('signal selection (true positive rate)')
    ax.set_title('ROC')
    ax.legend(["RNN: %.3f" % rnnScore, "classBDT: %.3f" % bdtScore], loc='best')
    plt.grid(True)
    fig.savefig("ROC_RNN_withH_allComb_withBTag_withClass_multiClass.png")
    plt.close(fig)
    #plt.show()
    return

def drawROC_pred(Y_test, p_test, weight_test, **kwargs):
    fig,ax = plt.subplots( nrows=1, ncols=1 )
    legend_text=[]
    fpr1, tpr1, thresholds1 = roc_curve(Y_test, p_test, sample_weight=weight_test)
    rnnScore = roc_auc_score(Y_test, p_test, sample_weight=weight_test)
    print("tree AUC on test: %f" % rnnScore)
    ax.plot(fpr1, tpr1)
    legend_text.append("tree: %.3f" % rnnScore)
    for key, value in kwargs.iteritems():
        fpr, tpr, thresholds = roc_curve(Y_test, value, sample_weight=weight_test)
        score = roc_auc_score(Y_test, value, sample_weight=weight_test)
        print("AUC on test %s: %f" % (key, score))
        ax.plot(fpr, tpr)
        legend_text.append("%s: %.3f" % (key, score))
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('bkg selection (false positive rate)')
    ax.set_ylabel('signal selection (true positive rate)')
    ax.set_title('ROC')
    ax.legend(legend_text, loc='best')
    plt.grid(True)
    fig.savefig("ROC_pred.png")
    plt.close(fig)
    #plt.show()
    return

def drawROC_check(Y, Y_1, Y_2, p, p_1, p_2, weight, weight_1, weight_2, class_id):
    lw=2
    fpr, tpr, t = roc_curve(Y[:, class_id], p[:, class_id], sample_weight=weight)
    roc_auc = auc(fpr, tpr, reorder=True)

    fpr1, tpr1, t1 = roc_curve(Y_1[:, class_id], p_1[:, class_id], sample_weight=weight_1)
    roc_auc_1 = auc(fpr1, tpr1, reorder=True)

    fpr2, tpr2, t2 = roc_curve(Y_2[:, class_id], p_2[:, class_id], sample_weight=weight_2)
    roc_auc_2 = auc(fpr2, tpr2, reorder=True)

    fig,ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(fpr, tpr, lw=lw,
        label='ROC curve of tot (area = {0:0.3f})'.format(roc_auc))

    ax.plot(fpr1, tpr1, lw=lw,
        label='ROC curve of fold-1 (area = {0:0.3f})'.format(roc_auc_1))

    ax.plot(fpr2, tpr2, lw=lw,
        label='ROC curve of fold-2 (area = {0:0.3f})'.format(roc_auc_2))

    
    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlabel('bkg selection (false positive rate)')
    ax.set_ylabel('signal selection (true positive rate)')
    ax.set_title('ROC')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right")

    plt.grid(True)
    fig.savefig("ROC_foldCheck_"+str(class_id)+".png")
    plt.close(fig)
    #plt.show()
    return

