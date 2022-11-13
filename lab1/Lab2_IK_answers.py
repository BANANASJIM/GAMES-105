import numpy as np
from scipy.spatial.transform import Rotation as R

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path,path_name,path1,path2 = meta_data.get_path_from_root_to_end()
    joint_parents = meta_data.joint_parent
    joint_initial_positions = meta_data.joint_initial_position
    joint_names = meta_data.joint_name
    temp_joint_positions = joint_positions
    temp_joint_orientations = joint_orientations
    root_joint_index = joint_names.index(meta_data.root_joint)
    end_joint_index = joint_names.index(meta_data.end_joint)
    # 末端位置
    end_position = temp_joint_positions[end_joint_index]
    path1 = list(reversed(path1))
    
    temp_joint_rotations =  np.tile(np.array([0.,0.,0.,1.0]),[len(joint_names),1])

    MAX_ITERATION = 50
    for it in range(MAX_ITERATION):
        prev_joint_rotations = temp_joint_rotations
        temp_joint_rotations =  np.tile(np.array([0.,0.,0.,1.0]),[len(joint_names),1])
        if np.linalg.norm(target_pose - end_position) < 0.01:
            break
        for i in range(len(path1)):
            
            joint_position = temp_joint_positions[path1[i]]
            joint_to_target = target_pose - joint_position
            joint_to_end =  end_position - joint_position
            
            rotation_angel = np.dot(joint_to_end,joint_to_target)
            rotation_angel = rotation_angel / (np.linalg.norm(joint_to_end)*np.linalg.norm(joint_to_target))
            rotation_angel = np.arccos(rotation_angel)
            rotation_vector = np.cross(joint_to_end,joint_to_target)
            normalized_vec = rotation_vector/np.linalg.norm(rotation_vector)
            
            rotation = R.from_rotvec(rotation_angel * normalized_vec)
            
            
            if joint_names[path1[i]][-4:] != "_end":
                temp_joint_rotations[path1[i]] = rotation.as_quat()
                end_position = joint_position + rotation.apply(joint_to_end)
            else:
                end_position = joint_position
                
                
        for index in range(len(joint_names)):
            #对根节点不做处理
            if index == root_joint_index:
                continue
            joint_rotation = R.from_quat(temp_joint_rotations[index])
            parent_orientation =R.from_quat(temp_joint_orientations[joint_parents[index]])
            parent_position = temp_joint_positions[joint_parents[index]]
            joint_offset = joint_initial_positions[index] - joint_initial_positions[joint_parents[index]]
            prev_joint_rotation = R.from_quat(prev_joint_rotations[index])
            joint_orientation = parent_orientation*prev_joint_rotation*joint_rotation
            
            temp_joint_orientations[index] =joint_orientation.as_quat()
            temp_joint_positions[index] =  parent_position + parent_orientation.apply(joint_offset)
        
    
    joint_positions = temp_joint_positions
    joint_orientations = temp_joint_orientations
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations