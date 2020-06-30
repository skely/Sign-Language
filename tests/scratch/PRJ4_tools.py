import numpy as np
# import LSTM_tools
# import matplotlib.pyplot as plt


def sign_synthesis(_sign_1, _sign_2, _gap_length, _type):
    """
    Synthesizes linear interpolation of movement
    :param _sign_1: trajectory frames1 X markers
    :param _sign_2: trajectory frames2 X markers
    :param _gap_length: number of frames (int)
    :param _type: type of interpolation: 'linear', 'kubic'
    :return: resulting_trajectory framesR X markers (dim = frames1+frames2+_gap_length X markers)
    """
    if _type == 'linear':
        _sign_1_t = _sign_1[-1:, :]
        _sign_2_t = _sign_2[0:1, :]
        inter = np.zeros((_gap_length, np.size(_sign_1_t, 1)))
        for traj in range(np.size(_sign_1_t, 1)):
            for frame in range(_gap_length):
                inter[frame, traj] = -1*(_sign_1_t[0, traj]-_sign_2_t[0, traj])/_gap_length*frame+_sign_1_t[0, traj]
        res = inter
    elif _type == 'kubic':
        if np.size(_sign_1, 0) < 2:
            res = -1
        else:
            _sign_1_t = _sign_1[-2:, :]
            _sign_2_t = _sign_2[0:2, :]
            inter = np.zeros((_gap_length, np.size(_sign_1_t, 1)))

            for traj in range(np.size(_sign_1_t, 1)):
                y1 = _sign_1_t[1, traj]
                y2 = _sign_2_t[0, traj]
                k1 = _sign_1_t[1, traj]-_sign_1_t[0, traj]
                k2 = _sign_2_t[1, traj]-_sign_2_t[0, traj]
                a = k1*_gap_length - (y2-y1)
                b = -k2*_gap_length + (y2-y1)

                for frame in range(_gap_length):
                    t = frame/_gap_length
                    inter[frame, traj] = (1-t)*y1 + t*y2 + t*(1-t)*((1-t)*a + t*b)
            res = inter
    else:
        res = -1
    return res[1:,:]


def sign_velocity_acceleration(_sign):
    """
    Calculates velocity and acceleration in the sign
    :param _sign: trajectory frames X markers
    :return: two vectors (length = frames): velocity, acceleration
    Uses the right difference: v(t) = x(t) - x(t+1)
    """

    if np.size(_sign, 0) < 3:
        return -1
    velocity = np.zeros((np.size(_sign, 0)-1, int(np.size(_sign, 1)/3)))
    acceleration = np.zeros((np.size(_sign, 0)-2, int(np.size(_sign, 1)/3)))

    for v in range(np.size(velocity, 0)):
        for traj in range(int(np.size(velocity, 1))):
            velocity[v, traj] = np.sqrt(np.power(_sign[v, 3*traj]-_sign[v+1, 3*traj], 2)+np.power(_sign[v, 3*traj+1]-_sign[v+1, 3*traj+1], 2)+np.power(_sign[v, 3*traj+2]-_sign[v+1, 3*traj+2], 2))

    for a in range(np.size(acceleration, 0)):
        for traj in range(np.size(acceleration, 1)):
            acceleration[a, traj] = (velocity[a, traj]-velocity[a+1, traj])

    return velocity, acceleration
    # plt.figure()
    # plt.plot(velocity)
    # plt.title('Velocity')
    # print(np.shape(velocity))
    # plt.figure()
    # plt.plot(acceleration)
    # plt.title('Acceleration')
    # print(np.shape(acceleration))


def max_velocity_acceleration(_sign):
    """
    Calculates max velocity and acceleration in the sign
    :param _sign: trajectory frames X markers
    :return: max_velocity, max_acceleration, arg_max_vel, arg_max_acc (type = float, float, int, int)
    """

    velocity, acceleration = sign_velocity_acceleration(_sign)
    max_velocity = np.amax(abs(velocity))
    max_acceleration = np.amax(abs(acceleration))
    arg_max_vel = np.argmax(np.amax(abs(velocity), axis=1))
    arg_max_acc = np.argmax(np.amax(abs(acceleration), axis=1))

    return max_velocity, max_acceleration, arg_max_vel, arg_max_acc


def compare_velocity(_sign_1, _sign_2):
    """
    Compares max velocity and acceleration of two signs
    :param _sign_1: trajectory frames X markers
    :param _sign_2: trajectory frames X markers
    :return: velocity_difference, acceleration_difference
    """
    velocity_difference = sign_velocity_acceleration(_sign_1)[0]-sign_velocity_acceleration(_sign_2)[0]
    acceleration_difference = sign_velocity_acceleration(_sign_1)[1]-sign_velocity_acceleration(_sign_2)[1]
    return velocity_difference, acceleration_difference


# def sign_error(_original, _approximation, _type_error='relative', _type_return='vector', _type_summarize='avg'):
#     """
#     Calculates the error between two signs
#     :param _original: trajectory frames X markers
#     :param _approximation: trajectory frames X markers
#             _sign_1 and _sign_2 must be same length
#     :param _type_error: type of error:  'absolute': {sum(i->N)[abs(t1_i-t2_i)]}
#                                         'relative': {sum(i->N)[abs((t1_i-t2_i)/t1_i)]}
#                                         'MSE': {sum(i->N)[(t1_i-t2_i)^2]}
#     :param _type_return: type of return:'total' (type = float) for a sum of errors
#                                         'vector' (type = numpy.ndarray) for a vector of errors for each frame
#     :param _type_summarize: type of summarize of errors of individual trajectories: 'sum', 'avg'
#     :return: error(based on params)
#     """
#     if np.shape(_original) != np.shape(_approximation):
#         return -1
#     if _type_summarize != 'sum' and _type_summarize != 'avg':
#         return -1
#     else:
#         if _type_error == 'absolute':
#             error = np.zeros(np.size(_original, 0))
#             for frame in range(np.size(error)):
#                 error[frame] = np.sum(abs(_approximation[frame, :]-_original[frame, :]))
#                 if _type_summarize == 'avg':
#                     error[frame] = error[frame]/np.size(_original, 1)
#             if _type_return == 'total':
#                 error = np.sum(error)
#         elif _type_error == 'relative':
#             _original_norm, orig_minmax = LSTM_tools.normalize(np.expand_dims(_original, axis=0))
#             _approximation_norm, _approx_minmax = LSTM_tools.normalize(np.expand_dims(_approximation, axis=0))
#             _original_norm = _original_norm[0, :, :]
#             _approximation_norm = _approximation_norm[0, :, :]
#             error = np.zeros(np.size(_original_norm, 0))
#
#             if _type_return == 'vector':
#                 for frame in range(np.size(error)):
#                     sum_orig = np.sum(_original_norm[frame, :])
#                     sum_approx = np.sum(_approximation_norm[frame, :])
#                     if _type_summarize == 'avg':
#                         sum_orig = sum_orig/np.size(_original, 1)
#                         sum_approx = sum_approx/np.size(_original, 1)
#                     error[frame] = (sum_orig-sum_approx)/sum_orig
#             elif _type_return == 'total':
#                 sum_orig = np.sum(_original_norm)
#                 sum_approx = np.sum(_approximation_norm)
#                 if _type_summarize == 'avg':
#                     sum_orig = sum_orig/np.size(_original, 1)
#                     sum_approx = sum_approx/np.size(_original, 1)
#                 error = (sum_orig-sum_approx)/sum_orig
#             else:
#                 error = -1
#         elif _type_error == 'MSE' or _type_error == 'mse':
#             error = np.zeros(np.size(_original, 0))
#             for frame in range(np.size(error)):
#                 error[frame] = np.sum(np.power(_approximation[frame, :] - _original[frame, :], 2))
#                 if _type_summarize == 'avg':
#                     error[frame] = error[frame]/np.size(_original, 1)
#             if _type_return == 'total':
#                 error = np.sum(error)/np.size(error)
#         else:
#             error = -1
#
#     return error
#