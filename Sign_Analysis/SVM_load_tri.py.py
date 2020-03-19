import numpy as np
import json
import TRC_tools
from sklearn import svm
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random

# fi = 'C:/Škola/FAV/PRJ/PRJ5/dictionary_takes_v3.txt'# '/home/jedle/data/Sign-Language/slovnik/pocasi_slovnik9.txt'

with open('D:/Škola/FAV/PRJ/BC/Output/trigrams.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    trigrams = pickle.load(f)

trigrams_valid = []
with open('D:/Škola/FAV/PRJ/BC/Output/trigrams.csv', 'w') as f:
    for item in trigrams:
        # print(item)
        if item[3] == -1:
            continue
        trigrams_valid.append([item[0]['domination_arm'], item[1]['domination_arm'], item[2]['domination_arm'],item[3]])

        # f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(item[0]['sign_name'], item[1]['sign_name'], item[2]['sign_name'],
        #                                      item[0]['hand_count'], item[1]['hand_count'], item[2]['hand_count'],
        #                                      item[0]['domination_arm'], item[1]['domination_arm'], item[2]['domination_arm'],
        #                                                  item[3]))

print((trigrams_valid))
mcs = []
for imc in range(10000):
    (random.shuffle(trigrams_valid))

    trigrams_valid_train = []
    trigrams_teacher_train = []
    trigrams_valid_test = []
    trigrams_teacher_test = []

    for sign in trigrams_valid[:int(len(trigrams_valid)*0.9)]:
        trigrams_valid_train.append(sign[0:-1])
        trigrams_teacher_train.append(sign[3])
    for sign in trigrams_valid[int(len(trigrams_valid)*0.9):]:
        trigrams_valid_test.append(sign[0:-1])
        trigrams_teacher_test.append(sign[3])

    svm_trigram = svm.LinearSVC()
    svm_trigram.fit(trigrams_valid_train, trigrams_teacher_train)

    correct = sum(svm_trigram.predict(trigrams_valid_test) == trigrams_teacher_test)
    mcs.append(correct/len(trigrams_valid_test))
print(len(trigrams_valid_test))
plt.hist(mcs)
print(plt.hist(mcs))
plt.show()
# xdd = []
# xddd = []
# svm_trigram = svm.LinearSVC()
# for sign in trigrams_valid:
#     xdd.append(sign[0:-1])
#     xddd.append(sign[3])
# svm_trigram.fit(xdd, xddd)
# print(sum(svm_trigram.predict(xdd) == xddd))
# print(sum(svm_trigram.predict(xdd) == xddd)/len(xdd))