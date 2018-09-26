import matplotlib.pyplot as plt

# # summarize history for accuracy
def mor_training_acc(name, history):
    fig,ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(history.history['acc'])
    ax.plot(history.history['val_acc'])
    ax.set_title('model accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(['train(0.8)', 'val(0.2)'], loc='best')
    fig.savefig("monitor_acc_"+name+".eps")
    plt.close(fig)
    #plt.show()

