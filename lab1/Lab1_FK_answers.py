import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data

def find_joint_orientation_data(joint_name_list , joint_name, motion_data):
    """
    输入： 名称列表, joint的名称, 对应的motion data
    输出:
        joint_orientation_data: 返回欧拉角的List
    """
    joint_orientation_data = None
    
    # site end 剔除
    temp_name_list = joint_name_list
    for name in temp_name_list:
        if name[-4:] == "_end":
            temp_name_list.remove(name)
                   
    joint_name_index = temp_name_list.index(joint_name)
    #忽略根节点位移 
    joint_orientation_data = motion_data[... , 3 * (joint_name_index + 1)  : 3 * (joint_name_index + 1) + 3]
    return joint_orientation_data
    

def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = None

    name_stack = []
    temp_joint_offset = []
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            currentLine = lines[i].split()
            if currentLine[0] == 'ROOT':
                joint_name.append(currentLine[1])
                joint_parent.append(-1)

            if currentLine[0] == "{":
                name_stack.append(joint_name[-1])

            if currentLine[0] == "OFFSET":
                offset_tuple = (float(currentLine[1]), float(
                    currentLine[2]), float(currentLine[3]))
                temp_joint_offset.append(offset_tuple)

            if currentLine[0] == "JOINT":
                joint_name.append(currentLine[1])
                parent_name = name_stack[-1]
                joint_parent.append(joint_name.index(parent_name))

            if currentLine[0] == "}":
                name_stack.pop()

            if currentLine[0] == "End":
                joint_end_name = name_stack[-1] + "_end"
                joint_name.append(joint_end_name)
                parent_name = name_stack[-1]
                joint_parent.append(joint_name.index(parent_name))

        joint_offset = np.asarray(temp_joint_offset)

    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
    """
    joint_positions = None
    joint_orientations = None

    temp_joint_positions = []
    temp_joint_orientations = []

    current_frame_motion_data = motion_data[frame_id]

    # 对Root处理
    temp_joint_positions.append((float(current_frame_motion_data[3 * 0 + 0]),
                                 float(current_frame_motion_data[3 * 0 + 1]),
                                 float(current_frame_motion_data[3 * 0 + 2])))
    temp_joint_orientations.append(R.from_euler('XYZ',
                                                [float(current_frame_motion_data[3 * 1 + 0]),
                                                 float(
                                                     current_frame_motion_data[3 * 1 + 1]),
                                                    float(current_frame_motion_data[3 * 1 + 2])],
                                                degrees=True))

    site_end_count = 0
    for index in range(len(joint_name)):

        # 对Root上面单独处理
        if index == 0:
            continue

        # 剩下都是欧拉旋转值
        # 需要对 site end 特殊处理
        if joint_name[index][-4:] == "_end":
            site_end_count += 1
        rotation_index = 3 * (index + 1 - site_end_count) + 0

        current_rotation = R.from_euler('XYZ',
                                        [float(current_frame_motion_data[rotation_index + 0]),
                                         float(
                                             current_frame_motion_data[rotation_index + 1]),
                                         float(current_frame_motion_data[rotation_index + 2])],
                                        degrees=True)
        current_joint_offset = joint_offset[index]
        current_joint_prarent_position = temp_joint_positions[joint_parent[index]]
        current_joint_prarent_orientation = temp_joint_orientations[joint_parent[index]]

        current_joint_position = current_joint_prarent_position + \
            current_joint_prarent_orientation.apply(current_joint_offset)
        temp_joint_positions.append(current_joint_position)
        current_joint_orientation = current_joint_prarent_orientation * current_rotation
        temp_joint_orientations.append(current_joint_orientation)

    temp_quat_list = []
    for rotation in temp_joint_orientations:
        temp_quat_list.append(rotation.as_quat())

    joint_positions = np.asarray(temp_joint_positions)
    joint_orientations = np.asarray(temp_quat_list)

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出:
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
    """
    motion_data = None
    A_pose_motion_data = load_motion_data(A_pose_bvh_path)
    A_pose_joint_name, _ , _ = part1_calculate_T_pose(A_pose_bvh_path)
    T_pose_joint_name, _ , _ = part1_calculate_T_pose(T_pose_bvh_path)

    # site end 剔除
    temp_name_list = T_pose_joint_name
    for name in temp_name_list:
        if name[-4:] == "_end":
            temp_name_list.remove(name)
            
    for i in range(len(temp_name_list)):
        #填入根位移与朝向
        if i == 0 :
            motion_data = A_pose_motion_data[... , i : i + 6]
            continue
        
        T_joint_name = temp_name_list[i]
        A_joint_orientation = find_joint_orientation_data(A_pose_joint_name,T_joint_name,A_pose_motion_data)
        if T_joint_name == "lShoulder":
            A_joint_orientation += np.array([0,0,-45])
        if T_joint_name == "rShoulder":
            A_joint_orientation += np.array([0,0,45])
                
        motion_data = np.concatenate((motion_data , A_joint_orientation),axis = 1)
    
    return motion_data
