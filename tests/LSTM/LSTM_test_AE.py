import os
import datetime
import LSTM_train


if __name__ == '__main__':
    ref_time = datetime.datetime.now()
    set_epochs = [1000, 2000, 3000]
    set_kernels = [200, 300, 400]
    experiment = 'test_glo_v3'
    source_dir = '/home/jedle/data/Sign-Language/_source_clean/testing'
    data_dir = os.path.join(source_dir, experiment)
    prepared_data_file = os.path.join(source_dir, 'prepared_data_glo_30-30ns.npz')
    logfile = os.path.join(data_dir, 'losses.txt')
    NN_type = 'simple_autoencoder'

    for k in set_kernels:
        for e in set_epochs:
            history = getattr(LSTM_train, NN_type)(prepared_data_file, data_dir, k, e)
            akt_result = [{'ref_time': ref_time, 'NN_version': NN_type, 'kernels' : k, 'epochs' : e, 'loss' : history.history['loss'][-1], 'val_loss' : history.history['val_loss'][-1]}]
            LSTM_train.make_log(akt_result, logfile)