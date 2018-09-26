import uproot
import numpy as np
import data_tool
from rootpy.vector import LorentzVector
"""
Data preparation.
Need to be better organized. Should learn more from G.Louppe's macro
"""


def data_prepare(sig_file, bkg_file, var_order, var_obt, **kwargs):
    """
    Input: 
    sig_file, bkg_file: file path for signal and bkg. 
    var_order: wanted input features
    var_obt: other variables expected, in addition to X(feature matrix), Y(truth label matix), eventNumber, weight, sample_weight (abs. of weight)
    **kwargs: additional requirement to slim input samples: {'variable': 'requirement'}. e.g. {'nBtag_85':'>=4'}
    return:
    The prepared arrays for sig and bkg separately. e.g. sig_d = {'X': X, 'Y':Y, ...}, bkg_d = {'X': X, 'Y':Y, ...}
    """

    tth_file = sig_file   #"/home/guo/PycharmProjects/data/ClassificationForTom170614.root"
    ttbar_file = bkg_file #"/home/guo/PycharmProjects/data/ClassificationForTom170614.root" 

    nb_combinations=12
    varOrder = var_order

    #----------------- ttH ---------------------
    tree_sig = uproot.open(tth_file)["tth/nominal/EventVariableTreeMaker_6ji4bi_cont85/EventVariables"]
    eventWeight = tree_sig["eventWeight"].array()
    n_var=len(varOrder)
    n_evt_sig=len(eventWeight)
    #print n_evt_sig
    
    best_list=[]
    
    for i in range(nb_combinations):
        best=np.zeros((n_var, n_evt_sig)) #float digit more than root, TTHReco_best_leptop_mass entry[0]:2.97625594e+05, but in root 297625.59 ???
        for var_j, j in zip(varOrder, range(len(varOrder))):
            
            if (i==0):
                best[j]=tree_sig[var_j].array()
                print var_j
            else:
                var_jj=np.char.replace(var_j, 'best', 'best'+str(i))
                print var_jj
                best[j]=tree_sig[var_jj].array()
                
        best=best.T
        best_list.append(best)
                
    #--------------------------------------------------------------------------------------
    X_sig = np.copy(best_list[0])

    i=1
    while(i<12):
        X_sig = np.hstack((X_sig, best_list[i]))
        i += 1

    X_sig = X_sig.reshape((n_evt_sig,12,33))

    print("signal X shape: ")
    print X_sig.shape
    #print X_sig
    
    
    #-------------------------------------------------------------------
    
    weight_bTagSF_Continuous = tree_sig["weight_bTagSF_Continuous"].array()
    weight_sig = eventWeight*weight_bTagSF_Continuous #weight.shape = (347831,)
    sample_weight_sig = np.absolute(weight_sig)
    eventNumber_sig = tree_sig["eventNumber"].array() #eventNumber.shape = (347831,)
    Y_sig = np.ones(n_evt_sig)
    #Y_sig[:, 0] = 1.
    
    sig_obt_dict={}
    sig_obt_dict['eventNumber']   = np.copy(eventNumber_sig)
    sig_obt_dict['weight']        = np.copy(weight_sig)
    sig_obt_dict['sample_weight'] = np.copy(sample_weight_sig)
    sig_obt_dict['X']             = np.copy(X_sig)
    sig_obt_dict['Y']             = np.copy(Y_sig)

    for istring in var_obt:
        sig_obt_dict[istring] = tree_sig[istring].array()

        
    #---------------------------------------------
    #----------------- ttbar ---------------------
    tree_bkg = uproot.open(ttbar_file)["tt_new/nominal/EventVariableTreeMaker_6ji4bi_cont85/EventVariables"]
    eventWeight = tree_bkg["eventWeight"].array()
    #print("Best: initialize the (%d, %d) array to save the training input. %d features, %d events" % (a1, a2, a1, a2))
    n_evt_bkg=len(eventWeight)
    
    best_list=[]
    for i in range(nb_combinations):
        best=np.zeros((n_var, n_evt_bkg)) #float digit more than root, TTHReco_best_leptop_mass entry[0]:2.97625594e+05, but in root 297625.59 ???
        for var_j, j in zip(varOrder, range(len(varOrder))):

            if (i==0):
                best[j]=tree_bkg[var_j].array()
                print var_j
            else:
                var_jj=np.char.replace(var_j, 'best', 'best'+str(i))
                print var_jj
                best[j]=tree_bkg[var_jj].array()

        best=best.T
        best_list.append(best)

    #------------------------------------------------------------------
    #X_bkg = np.hstack((best, best1, best2, best3, best4, best5, best6, best7, best8, best9, best10, best11))
    X_bkg = np.copy(best_list[0])

    i=1
    while(i<12):
        X_bkg = np.hstack((X_bkg, best_list[i]))
        i += 1


    X_bkg = X_bkg.reshape((n_evt_bkg,12,33))
    print("bkg X shape: ")
    print X_bkg.shape

    #-------------------------------------------------------------------
    weight_bTagSF_Continuous = tree_bkg["weight_bTagSF_Continuous"].array()
    weight_bkg = eventWeight*weight_bTagSF_Continuous #weight.shape = (347831,)
    sample_weight_bkg = np.absolute(weight_bkg)
    eventNumber_bkg = tree_bkg["eventNumber"].array() #eventNumber.shape = (347831,)
    Y_bkg = np.zeros(n_evt_bkg)
    #Y_bkg[:, 1] = 1
    
    bkg_obt_dict={}

    bkg_obt_dict['eventNumber']   = np.copy(eventNumber_bkg)
    bkg_obt_dict['weight']        = np.copy(weight_bkg)
    bkg_obt_dict['sample_weight'] = np.copy(sample_weight_bkg)
    bkg_obt_dict['X']             = np.copy(X_bkg)
    bkg_obt_dict['Y']             = np.copy(Y_bkg)

    for istring in var_obt:
        bkg_obt_dict[istring] = tree_bkg[istring].array()

    print ("Obtained variables for sig: %s" % sig_obt_dict.keys() )
    print ("Obtained variables for bkg: %s" % bkg_obt_dict.keys() )

    if(kwargs): #kwargs:{'nBtag_85': '>=4'}
        for var, n_cut in kwargs.iteritems():
            tx_sig = tree_sig[var].array()
            index_sig = eval("tx_sig" + n_cut)
            for key, value in sig_obt_dict.iteritems():
                sig_obt_dict[key] = value[index_sig]
            #print("Sig# after filtering b-tagged jet number: %d" % len(sig_obt_dict['X'] ))

            tx_bkg = tree_bkg[var].array()
            index_bkg = eval("tx_bkg" + n_cut)
            for key, value in bkg_obt_dict.iteritems():
                bkg_obt_dict[key] = value[index_bkg]
            #print("Bkg# after filtering b-tagged jet number: %d" % len(bkg_obt_dict['X'] ))                           

    return sig_obt_dict, bkg_obt_dict

def merge_sig_bkg(sig_obt_dict, bkg_obt_dict, do_debug=False):
    #---------- Merge signal and background -------------
    obt_dict={}
    n_evt = len(sig_obt_dict['weight']) + len(bkg_obt_dict['weight'])
    randomize = np.arange(n_evt)
    np.random.shuffle(randomize)

    for key in bkg_obt_dict.keys():
        sig_var = sig_obt_dict[key]
        bkg_var = bkg_obt_dict[key]
        merge_var = np.concatenate((sig_var, bkg_var), axis=0)
        merge_var = merge_var[randomize]
        if(do_debug):
            if(key=='X'):
                print("Debug mode: only 1000 events are obtained.")
            merge_var = merge_var[:10000]
        obt_dict[key] = merge_var
     
    print ("Merging sig and bkg samples! \nEvents are shuffled! \nObtained merged variables: %s" % obt_dict.keys() )
    return obt_dict


def match_filter(m_class='signal'):
    """
    A closure.
    For tth, filter the truth matching
    For ttbar, filter the 12th matching
    Need to do filter for tth and ttbar separately
    """
    def do_filter(obt_d):
        """
        Decorate the input feature arrays
        """
        if (m_class == 'signal'):
            #First ravel all combinations for tth
            X_tmp = np.copy(obt_d['X'])
            a0, a1, a2 = X_tmp.shape
            print a0, a1, a2
            X_new = X_tmp.reshape((a0*a1, a2), order='F')
            
            tmp = X_new[:, 0].astype(int)
            index_truth = ((tmp & 15)==15) & ((tmp & 48)!=0)
            
            for key, value in obt_d.iteritems():
                if(key=="X"):
                    X_new = np.delete(X_new, 0,1)
                    obt_d[key] = X_new[index_truth]
                else:
                    var_t = value.tolist()*12
                    var_t = np.array(var_t)
                    obt_d[key] = var_t[index_truth]
            print("Truth matchings of ttH are sustained as events. Feature matrix shape: %d, %d" % obt_d['X'].shape)
        elif(m_class == 'background'):
            obt_d['X'] = np.delete(obt_d['X'][:, 11, :], 0, 1)
            print("12th matching of ttbar is sustained as events. Feature matrix shape: %d, %d" % obt_d['X'].shape)

    return do_filter


def lorentz_trans(obt_d):
    """
    Assume that X[ievent] only contains (in order) pt, eta, phi, mass
    """
    #After chacking the functionality of this function, the matchpattern feature should also be removed.
    a0, a1 = obt_d['X'].shape
    X_new = np.zeros((a0, a1))
    for i, ix in zip(range(a0), obt_d['X']):
        for j in range(8): #8 objects in the game
            tmp = LorentzVector()
            tmp.set_pt_eta_phi_e(ix[4*j + 0], ix[4*j + 1], ix[4*j + 2], ix[4*j + 3])
            X_new[i][4*j + 0] = tmp.px
            X_new[i][4*j + 1] = tmp.py
            X_new[i][4*j + 2] = tmp.pz
            X_new[i][4*j + 3] = tmp.e

    new_d = {}
    for key, value in obt_d.iteritems():
        new_d[key] = value
    
    new_d['X'] = X_new

    return new_d


def even_odd_split(arr, eventNumber):
    arr_even = arr[eventNumber % 2 == 0]
    arr_odd = arr[eventNumber % 2 == 1]

    return arr_even, arr_odd

def train_test_split(arr, train_ratio=0.9):
    arr_even = arr[:int( len(arr)*train_ratio )]
    arr_odd = arr[int( len(arr)*train_ratio ) :]

    return arr_even, arr_odd

def balance_class(sample_weight, Y):
    """
    Normalize all bkg categories to sig
    """
    if(len(Y.shape) == 1):
        w = sample_weight[Y==1].sum()/sample_weight[Y==0].sum()
        sample_weight[Y==0] = sample_weight[Y==0]*w
    elif(len(Y.shape) == 2):
        cat = Y.shape[1]
        for ic in range(1, cat):
            w = sample_weight[Y[:, 0]==1].sum()/sample_weight[Y[:, ic]==1].sum()
            sample_weight[Y[:, ic]==1] = sample_weight[Y[:, ic]==1]*w

    return sample_weight

"""
sig_file = '/home/guo/PycharmProjects/data/new_file.root'
bkg_file = sig_file
var_order = ["TTHReco_best_truthMatchPattern", "TTHReco_withH_best_b1Higgsmv2_pT", "TTHReco_withH_best_b1Higgsmv2_eta", "TTHReco_withH_best_b1Higgsmv2_phi", "TTHReco_withH_best_b1Higgsmv2_E"]

#Obtained variables: X, Y, eventNumber, sample_weight, weight (5 in total) are default obtained vars. List the additional expected vars here.
var_obt = ["ClassifBDTOutput_inclusive_withBTag_new"]

cut_d = {'nBTags_85':'>= 4'}
data_prepare(sig_file, bkg_file, var_order, var_obt, **cut_d)
"""
