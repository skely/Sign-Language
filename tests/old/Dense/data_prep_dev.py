import os
import numpy as np
import matplotlib.pyplot as plt

def double_size(_image):
    h, v, d = np.shape(_image)
    _large_image = np.zeros((h * 2, v * 2, d))
    for i in range(h):
        for j in range(v):
            _large_image[i * 2:i * 2 + 2, j * 2:j * 2 + 2, :] = _image[i, j, :]
    return _large_image


def half_size(_image):
    h, v, d = np.shape(_image)
    _small_image = np.zeros((int(h/2), int(v/2), d))
    for i in range(int(h/2)):
        for j in range(int(v/2)):
            for k in range(d):
                _small_image[i, j, k] = np.average(_image[i * 2:i * 2 + 2, j * 2:j * 2 + 2, k])
    return _small_image


def show_pic(_pic, show=False):
    plt.figure()
    if 'float' in str(type(_pic[0, 0, 0])):
        plt.imshow((_pic*256).astype(np.uint8))
    else:
        plt.imshow(_pic)
    if show:
        plt.show()



if __name__ == '__main__':
    path = '/home/jedle/data/places2/my_test'
    original_name = 'orig.png'
    mask_name = 'mask.png'
    original = plt.imread(os.path.join(path, original_name))[:, :, :3]
    mask_orig = plt.imread(os.path.join(path, mask_name))[:, :, :3]

    tested2 = double_size(original)
    mask2 = double_size(mask_orig)

    tested4 = double_size(tested2)
    mask4 = double_size(mask2)

    tested8 = double_size(tested4)
    mask8 = double_size(mask4)

    # show_pic(original)
    # show_pic(tested8, show=True)

    plt.imsave(os.path.join(path, 'in_image.png'), tested2)
    plt.imsave(os.path.join(path, 'in_mask.png'), mask2)

    # picture_in = plt.imread(os.path.join(path, 'orig_double.png'))[:,:,:3]
    # mask_in = plt.imread(os.path.join(path, 'mask_double.png'))[:,:,:3]
    # output = plt.imread(os.path.join(path, 'out_double.png'))[:,:,:3]
    # output2 = plt.imread(os.path.join(path, 'out2_double.png'))[:,:,:3]
    #
    #
    # print(np.shape(picture_in))
    # print(np.shape(mask_in))
    # print(np.shape(output))
    # print(np.shape(output2))
    #
    #
    # h,v,d = np.shape(picture_in)
    # picture_mod = np.zeros((h+2, v+2, d))
    # picture_mod[1:-1, 1:-1, :] = picture_in
    #
    #
    # mask_mod = np.zeros((h+2, v+2, d))
    # mask_mod[1:-1, 1:-1, :] = mask_in
    #
    # plt.imshow(picture_mod)
    # plt.figure()
    # plt.imshow(picture_in)
    # plt.figure()
    # plt.imshow(mask_mod)
    # plt.show()
    #
    # plt.imsave(os.path.join(path, 'orig_mod'), picture_mod)
    # plt.imsave(os.path.join(path, 'mask_mod'), mask_mod)
    #
