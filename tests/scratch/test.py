from lib import BVH, data_prep
import os
import numpy as np

source_dir = '/home/jedle/data/Sign-Language/_source_clean/'
bvh_dir = '/home/jedle/data/Sign-Language/_source_clean/bvh/'
bvh_src_file = '/home/jedle/data/Sign-Language/_source_clean/bvh/16_05_20_a_R.bvh'
take_dict_file = os.path.join(source_dir, 'dictionary_takes_v3.txt')
dict_dict_file = os.path.join(source_dir, 'dictionary_dict_v3.txt')

# dic = SL_dict.read_raw(take_dict_file)

selected_sign_id = 'sobota'  # transitions
selected_sign_name = 'tra.'
selected_file = '16_05_20_a_R.bvh'
selected_channels = 'rotation'  # kanály obsahující pouze rotace

m, c, _ = BVH.get_joint_list(bvh_src_file)
ch_sel_idxs = BVH.get_joint_id(m, c, '', _channel_name='rotation')
ret, meta = data_prep.mine_sign_trajectories(bvh_dir, take_dict_file, 10, _sign_id=selected_sign_id, _sign_name=selected_sign_name, _verbose=True)

for r, m in zip(ret, meta):
    print(np.shape(r[:, ch_sel_idxs]))
    print(m)
