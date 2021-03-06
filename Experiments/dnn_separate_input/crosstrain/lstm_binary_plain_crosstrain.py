import sys, os
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

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, LearningRateScheduler
from keras import optimizers
from keras.models import load_model

#Import my modules
sys.path.append(os.path.abspath('../../../'))
from my_data import prepare, prepro
from my_structures import model_lstm  as lunch_model
from my_draw import train_monitor
from my_callback.sd_callback import roc_cb_earlyStop
import math as m

"""
This is training macro of parse tree, with train on even events for the moment
"""

"""
More updates are needed:
#1. Check the correct way of monitoring training epoch.
2. Use a better callback function
"""

sig_file = '/data1/home/ziyu.guo/data/ClassificationForTom170614.root'
bkg_file = sig_file
var_order = ["TTHReco_withH_best_leptop_mass", "TTHReco_withH_best_hadtop_mass", "TTHReco_withH_best_hadW_mass",
             "TTHReco_withH_best_hadWblepTop_mass", "TTHReco_withH_best_lepWbhadTop_mass", "TTHReco_withH_best_hadWbhadTop_dR", 
             "TTHReco_withH_best_hadWblepTop_dR", "TTHReco_withH_best_lepblepTop_dR", "TTHReco_withH_best_lepbhadTop_dR",
             "TTHReco_withH_best_blepTopbhadTop_dR", "TTHReco_withH_best_qqhadW_dR", "TTHReco_withH_best_bhadTopq1hadW_dR", 
             "TTHReco_withH_best_bhadTopq2hadW_dR", "TTHReco_withH_best_minbhadTopqhadW_dR", "TTHReco_withH_best_diff_mindRbhadTopqhadW_dRlepblepTop",
             "TTHReco_withH_best_Higgs_mass", "TTHReco_withH_best_Higgsq1hadW_mass", "TTHReco_withH_best_bbHiggs_dR", 
             "TTHReco_withH_best_lepb1Higgs_dR", "TTHReco_withH_best_bhadTop_tagWeightBin", "TTHReco_withH_best_blepTop_tagWeightBin",
             "TTHReco_withH_best_b1Higgs_tagWeightBin", "TTHReco_withH_best_b2Higgs_tagWeightBin", "TTHReco_withH_best_qhadW_tagWeightBin_1", 
             "TTHReco_withH_best_qhadW_tagWeightBin_2", 
             "dRbb_avg_Sort4", "dRbb_MaxPt_Sort4", "dEtajj_MaxdEta", "Mbb_MindR_Sort4", "nHiggsbb30_Sort4", "Aplanarity_bjets_Sort4", "H1_all"]

#Obtained variables: X, Y, eventNumber, sample_weight, weight (5 in total) are default obtained vars. List the additional expected vars here.
var_obt = ["ClassifBDTOutput_inclusive_withBTag_new"]

cut_d = {'nBTags_85':'>= 4'}
sig_obt_dict, bkg_obt_dict = prepare.data_prepare(sig_file, bkg_file, var_order, var_obt, **cut_d)

# prepare.match_filter('signal')(sig_obt_dict)
# prepare.match_filter('background')(bkg_obt_dict)
    
data_dict = prepare.merge_sig_bkg(sig_obt_dict, bkg_obt_dict, do_debug = True)

#data_dict = prepare.lorentz_trans(data_dict)


##################### even-odd splitting, weight balancing #####################################
eventNumber = data_dict['eventNumber']
        
X_e, X_o = prepare.even_odd_split(data_dict['X'], eventNumber)
Y_e, Y_o = prepare.even_odd_split(data_dict['Y'], eventNumber)
weight_e, weight_o = prepare.even_odd_split(data_dict['weight'], eventNumber)
sample_weight_e, sample_weight_o = prepare.even_odd_split(data_dict['sample_weight'], eventNumber)
sample_weight_e = prepare.balance_class(sample_weight_e, Y_e)
sample_weight_o = prepare.balance_class(sample_weight_o, Y_o)

###############  FOLD-1  ###################
print("... ... FOLD-1: learn, validation on even, app on odd ... ...")
##### learn, validation, application spliting, scale and norm #####
val_split=0.2
X_e_learn, X_e_val = prepro.learn_val_split(X_e, val_split)
Y_e_learn, Y_e_val = prepro.learn_val_split(Y_e, val_split)
sample_weight_e_learn, sample_weight_e_val = prepro.learn_val_split(sample_weight_e, val_split)
weight_e_learn, weight_e_val = prepro.learn_val_split(weight_e, val_split)

X_e_learn, X_o_app, X_e_val, l_mean, l_std = prepro.scale_norm(X_e_learn, X_o, X_validation=X_e_val)
Y_o_app = np.copy(Y_o)
sample_weight_o_app = np.copy(sample_weight_o)
weight_o_app = np.copy(weight_o)

print("... ... # of events for train: %d, val: %d, test: %d" % (len(Y_e_learn), len(Y_e_val), len(Y_o_app) ) )

#---------------- fit model  -----------------------
model_1 = lunch_model.m_model(12,32)

m_adam = optimizers.Adam()#lr=0.482, decay=0.003)
model_1.compile(loss='binary_crossentropy', optimizer=m_adam, metrics=['accuracy'])
print model_1.summary()

input_X = X_e_learn
input_Y = Y_e_learn

val_X = X_e_val
val_Y = Y_e_val

filename = "Weights-even-odd-{epoch:02d}-{auc:.3f}.hdf"
callbacks_auc_1 = roc_cb_earlyStop(input_X, val_X, input_Y, val_Y, weight_e_learn, weight_e_val, filepath=filename)
callbacks= [# ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1),
            callbacks_auc_1]


history_1 = model_1.fit(input_X, input_Y, epochs=5, batch_size=200, sample_weight=sample_weight_e_learn, callbacks=callbacks
                        , validation_data=(val_X, val_Y, sample_weight_e_val) )

train_monitor.mon_training("model_even_odd", history_1, "loss")
train_monitor.mon_training("model_even_odd", history_1, "acc")
train_monitor.mon_auc("model_even_odd", callbacks_auc_1)


###############  FOLD-2  ###################
print("... ... FOLD-2: learn, validation on even, app on odd ... ...")
##### learn, validation, application spliting, scale and norm #####
val_split=0.2
X_o_learn, X_o_val = prepro.learn_val_split(X_o, val_split)
Y_o_learn, Y_o_val = prepro.learn_val_split(Y_o, val_split)
sample_weight_o_learn, sample_weight_o_val = prepro.learn_val_split(sample_weight_o, val_split)
weight_o_learn, weight_o_val = prepro.learn_val_split(weight_o, val_split)

X_o_learn, X_e_app, X_o_val, l_mean, l_std = prepro.scale_norm(X_o_learn, X_e, X_validation=X_o_val)
Y_e_app = np.copy(Y_e)
sample_weight_e_app = np.copy(sample_weight_e)
weight_e_app = np.copy(weight_e)

print("... ... # of events for train: %d, val: %d, test: %d" % (len(Y_e_learn), len(Y_e_val), len(Y_o_app) ) )

#---------------- fit model  -----------------------
model_2 = lunch_model.m_model(12,32)

m_adam = optimizers.Adam()#lr=0.482, decay=0.003)
model_2.compile(loss='binary_crossentropy', optimizer=m_adam, metrics=['accuracy'])
print model_2.summary()

input_X = X_o_learn
input_Y = Y_o_learn

val_X = X_o_val
val_Y = Y_o_val

filename = "Weights-odd-even-{epoch:02d}-{auc:.3f}.hdf"
callbacks_auc_2 = roc_cb_earlyStop(input_X, val_X, input_Y, val_Y, weight_o_learn, weight_o_val, filepath=filename)
callbacks= [# ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1),
            callbacks_auc_2]


history_2 = model_2.fit(input_X, input_Y, epochs=5, batch_size=200, sample_weight=sample_weight_o_learn, callbacks=callbacks
                        , validation_data=(val_X, val_Y, sample_weight_o_val) )

train_monitor.mon_training("model_odd_even", history_2, "loss")
train_monitor.mon_training("model_odd_even", history_2, "acc")
train_monitor.mon_auc("model_odd_even", callbacks_auc_2)
    
