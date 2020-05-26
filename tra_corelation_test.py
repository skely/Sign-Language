import os
from lib import bvh2glo_simple
import numpy as np
from lib import SL_dict
import matplotlib.pyplot as plt
from lib import dist_comp #přidaná funkce
import operator
import math
import glob

if __name__ == "__main__":
    done_files = 0

    # file_list = glob.iglob('C:/Users/User/PRJ4/data_bvh/*.bvh')
    path = '/home/jedle/data/Sign-Language/_source_clean/bvh'
    file_list = os.listdir(path)
    file_list = [f for f in file_list if ('17' in f or '16' in f)]
    # dictionary_file = 'C:/Users/User/PRJ4/data/ultimate_dictionary2.txt'
    dictionary_file = '/home/jedle/Projects/Sign-Language/data/ultimate_dictionary2.txt'

    for filepath in file_list: #iterování přes jednotlivé soubory
        if '16_10_25_a_R' in filepath or '17_01_15_b_FR' in filepath or '17_03_25_b_R' in filepath:
            continue
        BVH_file = os.path.join(path, filepath)

        #načtení BVH souboru a přepočítání angulárních dat na trajektorii v globáních souřadnicích
        #BVH_infile = '16_05_20_a_R.bvh'
        joints, trajectory = bvh2glo_simple.calculate(BVH_file)
        frames, joint_id, channels = np.shape(trajectory)

        ranges = {} #budoucí pole snímkových rozsahů jednotlivých transitions, např transition, která je ve slovníku jako 4 znak bude 4: počet snímků
        trajectories = [] #budoucí pole trajektorií všech částí těla během transitions
        number = 0
        DISTS = [] #budoucí pole vzdáleností, které ucestovaly jednotlivé části těla během transitions

        dictionary = SL_dict.search_take_file(dictionary_file, BVH_file)
        for line in dictionary:
            number += 1
            for tmp_key in line.keys():
                if line['sign_name'] == 'tra.':
                    rang = line['annotation_Filip_bvh_frame'] 

                    start = rang[0] #první číslo z annotation_Filip... je snímek na kterém tra. začne
                    end = rang[1] #naopak
                    DIST = dist_comp.comp(trajectory,start,end) #výpočet absolutní vzdálenosti částí těla mezi start a end implementovanou funkcí
                    if not DIST in DISTS:
                        DISTS.append(DIST)
                    ranges['{}'.format(number)] = end-start
                    trajectories.append(trajectory[number])
                    trajectories.append('******')

        Dist_sorted = [] #přeuspořádání DISTS, index řádku = část těla, obsahuje všechny polohy dané části těla ve snímcích, nutno pro použití ve funkci np.corrcoef()
        numberof = 0
        print(filepath)
        while numberof < len(DISTS[1]):
            Dist_sorted.append([])
            for i in range(len(DISTS)):
                Dist_sorted[numberof].append(DISTS[i][numberof])
            numberof +=1
        
        corr_coefs = [] #spočtení korelačních koeficientů změny polohy všech částí těla s délkou transitions
        for i in range(len(DISTS[1])):
            ranges_values = []
            for key in ranges:
                ranges_values.append(ranges[key])
            corr_matrix = np.corrcoef(Dist_sorted[i], ranges_values)
            corr_coefs.append(corr_matrix[0][1])

        avg_dist_singly = [] #průměrná uražená vzdálenost pro každou část těla
        for i in range(len(Dist_sorted)):
            avg_dist_singly.append(np.mean(Dist_sorted[i]))
        avg_dist_all = np.mean(avg_dist_singly) #celková průměrná uražená vzdálenost

        maxx = []
        indexes = []
        #avg_dists = []
        for i in range(3): #hledání a uložení 3 největších korelačních koeficientů
            indexes.append(corr_coefs.index(max(corr_coefs)))
            maxx.append(corr_coefs.pop(corr_coefs.index(max(corr_coefs))))
            #avg_dists.append(np.mean(Dist_sorted[indexes[i]]))

        # file = open("results.txt","a") #append mode do souboru s informacemi
        # file.write('\n********************************************************\n')
        # file.write(BVH_file)
        # file.write('\nBiggest correlation: '+str(joints[indexes[0]])+', val: '+str(maxx[0])+', avg trav. dist: '+str(avg_dist_singly[indexes[0]]))
        # file.write('\nSecond b. correlation: '+str(joints[indexes[1]])+', val: '+str(maxx[1])+', avg trav. dist: '+str(avg_dist_singly[indexes[1]]))
        # file.write('\nThird b. correlation: '+str(joints[indexes[2]])+', val: '+str(maxx[2])+', avg trav. dist: '+str(avg_dist_singly[indexes[2]])+'\nAvg trav. dist all: '+str(avg_dist_all))
        # file.close()
        done_files +=1
        print(done_files)
        print('Biggest correlation with:',joints[indexes[0]],', value:',maxx[0],', avg travelled dist:',avg_dist_singly[indexes[0]])
        print('Second biggest correlation with:',joints[indexes[1]],', value:',maxx[1],', avg travelled dist:',avg_dist_singly[indexes[1]])
        print('Third biggest correlation with:',joints[indexes[2]],', value:',maxx[2],', avg travelled dist:',avg_dist_singly[indexes[2]],'\nAvg travelled dist for all:',avg_dist_all)
        # plt.scatter(Dist_sorted[indexes[0]], ranges.values())
    # plt.show()
