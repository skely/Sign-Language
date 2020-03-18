import numpy as np
import os
import json
from lib import SL_dict

source_dir = '/home/jedle/data/Sign-Language/_source_clean/'
model_code = 'model_BI_AE_lstm200_ep3.txt'
in_file = os.path.join(source_dir, model_code)

with open(in_file, 'r') as f:
    history = json.load(f)

print(model_code.split())
for kee in history:
    print(kee, history[kee][-1])

