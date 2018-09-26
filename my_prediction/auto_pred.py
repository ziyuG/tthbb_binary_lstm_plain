import pickle
import glob

def get_best_model(name_pattern):
    f=glob.glob(name_pattern)
    fepoch = [int(i_f.split('-')[-2]) for i_f in f]
    pattern_rep = name_pattern.replace("*-*.hdf", "{:02}-*.hdf").format(max(fepoch))
    model_weight = glob.glob(pattern_rep)[0]

    max_epoch = max(fepoch)
    max_auc = float(model_weight[:model_weight.find('.hdf')].split('-')[-1])

    print("...... LOADED MODEL: %s" % model_weight)

    return model_weight, max_epoch, max_auc

def save_roc_app(fpr, tpr, threshold, val_ep_auc):
    file_fpr=open('file_fpr','w')
    pickle.dump(fpr, file_fpr)
    file_fpr.close()

    file_tpr=open('file_tpr','w')
    pickle.dump(tpr, file_tpr)
    file_tpr.close()

    file_threshold=open('file_threshold','w')
    pickle.dump(threshold, file_threshold)
    file_threshold.close()

    val_info = val_ep_auc 
    file_val=open('file_val','w')
    pickle.dump(val_info, file_val)
    file_val.close()
