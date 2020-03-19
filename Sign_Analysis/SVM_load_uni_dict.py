import numpy as np
import json
import TRC_tools
from sklearn import svm
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random

# fi = 'C:/Škola/FAV/PRJ/PRJ5/dictionary_takes_v3.txt'# '/home/jedle/data/Sign-Language/slovnik/pocasi_slovnik9.txt'

with open('D:/Škola/FAV/PRJ/BC/Output/uni_dict.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    signs, teacher = pickle.load(f)

unigrams_valid = []
# with open('D:/Škola/FAV/PRJ/BC/Output/unigrams.csv', 'w') as f:
for i in range(len(teacher)):
    unigrams_valid.append([signs[i], teacher[i]])

        # f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(item[0]['sign_name'], item[1]['sign_name'], item[2]['sign_name'],
        #                                      item[0]['hand_count'], item[1]['hand_count'], item[2]['hand_count'],
        #                                      item[0]['domination_arm'], item[1]['domination_arm'], item[2]['domination_arm'],
        #                                                  item[3]))

print((unigrams_valid))
nmc = 1000000
mcs = []
results = []
for imc in range(nmc):
    if (imc/nmc*100)%1 ==0:
        print('{}%'.format(100*imc/nmc))
    (random.shuffle(unigrams_valid))

    trigrams_valid_train = []
    trigrams_teacher_train = []
    trigrams_valid_test = []
    trigrams_teacher_test = []

    for sign in unigrams_valid[:int(len(unigrams_valid) * 0.9)]:
        trigrams_valid_train.append(sign[0])
        trigrams_teacher_train.append(sign[1])
    for sign in unigrams_valid[int(len(unigrams_valid) * 0.9):]:
        trigrams_valid_test.append(sign[0])
        trigrams_teacher_test.append(sign[1])

    svm_trigram = svm.LinearSVC()
    svm_trigram.fit(trigrams_valid_train, trigrams_teacher_train)

    correct = sum(svm_trigram.predict(trigrams_valid_test) == trigrams_teacher_test)
    # print(correct)
    mcs.append(correct/len(trigrams_valid_test))
    # results.append(trigrams_valid_test)
    # print(trigrams_valid_test)
print(len(trigrams_valid_test))
print(len(trigrams_valid_train))
# plt.hist(mcs)
print(plt.hist(mcs))
print(np.var(mcs))
print(np.mean(mcs))
plt.title('Unigrams Dictionary')
plt.show()