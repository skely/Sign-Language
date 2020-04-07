from lib import bvh2glo_simple
import matplotlib.pyplot as plt


def my_colored_plot(_joint):
    if any(n in _joint for n in ['Right']):
        m_color = 'r'
    elif any(n in _joint for n in ['Left']):
        m_color = 'b'
    else:
        m_color = 'g'

    if any(n in _joint for n in ['1', '2', '3']):  # prsty
        m_shape = '+'
    elif any(n in _joint for n in ['Shoulder']):
        m_shape = 's'
    elif any(n in _joint for n in ['ForeArm']):
        m_shape = 'p'
    elif any(n in _joint for n in ['Arm']):
        m_shape = 'v'
    else:
        m_shape = '*'

    return m_color, m_shape


def run():
    BVH_file = '/home/jedle/data/Sign-Language/_source_clean/bvh/16_05_29_c_FR.bvh'
    # frame_number
    frame_number = 1000
    # size of axes in graph
    unisize = 50

    joints, results = bvh2glo_simple.calculate(BVH_file)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', title='frame: {}'.format(frame_number))
    for i, joint in enumerate(joints):
        m_color, m_shape = my_colored_plot(joint)
        ax.scatter(results[frame_number, i, 0], results[frame_number, i, 1], results[frame_number, i, 2], label=joint,
                   color=m_color, marker=m_shape)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(-unisize, unisize)
    ax.set_ylim3d(-unisize, unisize)
    ax.set_zlim3d(-unisize, unisize)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    """
    Demonstrator: how to use bvh2glo_simple.py
    """
    run()