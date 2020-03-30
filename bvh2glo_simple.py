from BVwHacker import bvh
from lib import BVH
import numpy as np
import sys


def calculate(_in_file):
    trajectory = BVH.load_trajectory(_in_file)
    header = BVH.load_raw_header(_in_file)
    tree_structure = BVH.get_tree_structure_joint_list(header)
    joints, channels, offsets, _, _ = tree_structure
    skeleton = bvh.Skeleton(_in_file, 1)
    total_length = np.size(trajectory, 0)

    retval = np.zeros((total_length, len(joints), 3))
    for i in range(total_length):
        skeleton.updateFrame(i)
        for c, joint in enumerate(joints):
            j = skeleton.getJoint(joint)
            retval[i, c, :] = j.worldpos[0], j.worldpos[1], j.worldpos[2]
        sys.stdout.write('\rDataprep processing... {:.2f}% done.'.format(100 * (i + 1) / total_length))
    sys.stdout.write('\rDataprep processing... done.\n')
    return joints, retval