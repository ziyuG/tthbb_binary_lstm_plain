from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from sklearn.ensemble import BaggingClassifier
from keras.wrappers.scikit_learn import KerasClassifier
# Create first network with Keras
from keras.layers import Dense, Reshape, Activation, Dropout, LSTM, Input, merge

from keras.optimizers import RMSprop, Adamax, Adagrad, SGD

from sklearn.metrics import roc_auc_score, roc_curve


############################################
################ Define model ###############
def m_model(n_comb, n_feature, n_aux_feature):

    #deine model
    inputs = Input(shape=(n_comb, n_feature))
    lstm_out = LSTM(50, dropout_U=0.2, dropout_W=0.2)(inputs)

    auxiliary_input = Input(shape=(n_aux_feature,), name='aux_input')
    x = merge([lstm_out, auxiliary_input], mode='concat')

    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(input=[inputs, auxiliary_input], output=predictions)

    #print the model structure
    print(model.summary())
    return model

