import os
import numpy as np
from bvh_parser.bvh import bvh_parser as bp
from bvh_parser.bvh import bvh_tools2020 as bw
# import bvh_cut

bvh_file_name_Jedle = "/home/jedle/data/Sign-Language/ELG/take01.bvh"
# bvh_file_name_Pavel = "/home/jedle/data/Sign-Language/16_05_20_a.bvh"
out_bvh_file = 'out_cut4.bvh'
bvh_parser = bp.bvh(bvh_file_name_Jedle)
# selected_joint = ['RightHand', 'LeftHand']
selected_joint = 'Head'

new_parser = bw.bvh_cut(bvh_parser, selected_joint)
# new_parser = bvh_cut.bvh_cut(new_parser, )
bw.save_bvh(out_bvh_file, bvh_parser)

# bvh_parser.motions[0]
# bvh_parser.skeleton

# selected_joint = "RightHandIndex2"
# print(bvh_parser.skeleton[selected_joint])
# remove_stack = [selected_joint]
# removed_log = []
# while remove_stack != []:
#     tmp = remove_stack.pop()
#     removed_log.append(tmp)
#     for jk in bvh_parser.skeleton.keys():
#         if bvh_parser.skeleton[jk]['parent'] == tmp:
#             remove_stack.append(jk)
# print(removed_log)
#
# for i, frame in enumerate(bvh_parser.motions):
#     for item in frame[1]:
#
#         if item in removed_log:
#             remove_stack.append(item)
#             bvh_parser.motions[i][1].remove(item)
#     break
# print(remove_stack)
# print(bvh_parser.motions[0][1])