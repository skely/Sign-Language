from bvh_lib import bvh_reader as br

bvh_src_file = "/home/jedle/data/Sign-Language/ELG/take01.bvh"
reader = br.bvh(bvh_src_file)

# print(type(reader.meta_data))
# print(reader.meta_data.keys())

for it in reader.meta_data.keys():
    print(it, reader.meta_data[it])

