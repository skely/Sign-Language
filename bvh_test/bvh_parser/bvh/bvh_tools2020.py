def save_bvh(file_name, bvh_data):
    """
    uses variables "skeleton" and "motions" from class bvh
    """
    root = None
    keyword = 'parent'
    char_tab, new_line, char_brace_open, char_brace_close = '\t', '\n', '{', '}'
    stack = []
    text = []
    close_brace = 0
    lvl = -1
    joint_order = []
    text.append('HIERARCHY\n')

    for dk in bvh_data.skeleton.keys():
        if bvh_data.skeleton[dk][keyword] == root:
            stack.append([dk, 0])
            break
    while stack != []:
        end = False
        oldlvl = lvl
        tmp, lvl = stack.pop()
        braces_left = lvl - oldlvl
        if braces_left < 1:
            for i in range(oldlvl, lvl - 1, -1):
                text.append((i * char_tab + char_brace_close + new_line))
        for dk in bvh_data.skeleton.keys():
            if bvh_data.skeleton[dk][keyword] == tmp:
                stack.append([dk, lvl + 1])

        if bvh_data.skeleton[tmp][keyword] == root:
            text.append(lvl * char_tab + 'ROOT ' + tmp + new_line)
        elif ('Nub' in tmp) or ('Site' in tmp):
            text.append(lvl * char_tab + 'End Site' + new_line)
            end = True
        else:
            text.append(lvl * char_tab + 'JOINT ' + tmp + new_line)
        text.append(lvl * char_tab + char_brace_open + new_line)
        offs = bvh_data.skeleton[tmp]['offsets']
        chans = bvh_data.skeleton[tmp]['channels']
        text.append((lvl + 1) * char_tab + 'OFFSET ' + '{:.5f} {:.5f} {:.5f}'.format(*offs) + new_line)
        if not end:
            text.append((lvl + 1) * char_tab + 'CHANNELS ' + str(len(chans)) + ' ' + ' '.join(chans) + new_line)
        joint_order.append([tmp, bvh_data.skeleton[tmp]['channels']])

    for i in range(lvl, -1, -1):
        text.append(i * char_tab + char_brace_close + new_line)

    text.append('MOTION' + new_line)
    text.append('Frames: {}{}'.format(len(bvh_data.motions), new_line))
    text.append('Frame Time: {:.6f}{}'.format(bvh_data.motions[1][0], new_line))

    for i in range(len(bvh_data.motions)):
        line = ' '
        tmp_frame = bvh_data.motions[i][1]
        j = joint_order[1][0]
        for j in joint_order:
            for i in range(len(tmp_frame)):
                if j[0] == tmp_frame[i][0]:
                    line += '{:.4f} '.format(tmp_frame[i][2])
        line = line[:-1] + new_line
        text.append(line)

    with open(file_name, 'w') as f:
        f.writelines(text)

def bvh_cut(bvh_data, joint_name, bone_end=False):
    """
    Remove bone and it's child bones
    :param bvh_data: bvh_parser data
    :return:
    """
    if isinstance(joint_name, str):
        remove_stack = [joint_name]
    else:
        remove_stack = joint_name
    removed_log = []
    while remove_stack != []:
        tmp = remove_stack.pop()
        removed_log.append(tmp)
        for jk in bvh_data.skeleton.keys():
            if bvh_data.skeleton[jk]['parent'] == tmp:
                remove_stack.append(jk)
    for i, frame in enumerate(bvh_data.motions):
        tmp_frame = frame[1]
        new = [x for x in tmp_frame if x[0] not in removed_log]
        new_frame = (frame[0], new)
        bvh_data.motions[i] = new_frame
    for joint in removed_log:
        bvh_data.skeleton.pop(joint)


def new_bone(bvh_data, name, parent, offsets, channels, motions=None):
    new_bone = { "parent" : parent, "channels" : channels, "offsets" : offsets}
    if parent in bvh_data.skeleton:
        bvh_data.skeleton[name] = new_bone
        # ********** same length and time stamps check **********
        if motions is not None:
            if [x for x,y in zip(motions, bvh_data.motions) if x[0] != y[0]] == []:
            # *******************************************************
                for i in range(len(bvh_data.motions)):
                    tmp = list(bvh_data.motions[i])
                    tmp[1] = tmp[1] + (motions[i][1])
                    bvh_data.motions[i] = tuple(tmp)
            else:
                print('motion data does not match in length and time stamps.')
    else:
        print('parent does not exist.')

def copy_motion(bvh_data, bone_name):
    new_motions = []
    for i in range(len(bvh_data.motions)):
        matched_bones_motions = [bone_motions for bone_motions in bvh_data.motions[i][1] if bone_motions[0] == bone_name]
        new_motions.append([bvh_data.motions[i][0], matched_bones_motions])
    return new_motions