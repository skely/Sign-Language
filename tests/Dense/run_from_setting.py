import os
import numpy as np
import dense_testing

def read_settings(_infile, _settings):
    with open(_infile, 'r') as f:
        cont = f.readlines()
    response = {}
    for line in cont:
        line = line.strip()
        if '#' not in line:
            tmp = line.split('=')
            if len(tmp) > 1:
                if tmp[0] in settings:
                    response[tmp[0]] = tmp[1]
    return response


if __name__ == '__main__':
    settings_file = '/home/jedle/Projects/Sign-Language/tests/Dense/test_setting.txt'
    settings = ['architecture', 'depth', 'epochs', 'optimizer', 'learning_rate']

    rsp = read_settings(settings_file, settings)
    print(rsp)


    path = '/home/jedle/Projects/Sign-Language/tests/Dense'
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'

    data = dense_testing.prepare_data_file(data_file)

    loss = 'mean_squared_error'
    optimizer = adam(learning_rate=0.001)
    epochs = 100
    batch_size = 100
    hidden_layer_sizes = [[9], [27], [81], [248]]
    results = []
    repetitions = 3


    for h in hidden_layer_sizes:
        for r in range(repetitions):
            model_name = 'model_Dense_v1_{}_r{}'.format(h[0], r)
            model_s, accuracy_s = Dense_v1_3layer_noskip(data, loss, optimizer, epochs, batch_size, h)
            model_n, accuracy_n = Dense_v1_3layer(data, loss, optimizer, epochs, batch_size, h)
            results.append([h, r, accuracy_s, accuracy_n])
            model_s.save(os.path.join(path, model_name+'s'))
            model_n.save(os.path.join(path, model_name+'n'))

    print(results)
    np.save(os.path.join(path, 'results2'), results)
