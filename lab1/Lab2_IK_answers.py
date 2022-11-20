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
    temp_joint_positions = joint_positions.copy()
    
    temp_joint_orientations = joint_orientations.copy()
    
    root_joint_index = joint_names.index(meta_data.root_joint)
    end_joint_index = joint_names.index(meta_data.end_joint)

    joint_dists = []
    dist = 0
    
    for i in range(len(path)-1):
        joint_dist = np.linalg.norm(joint_initial_positions[path[i]] - joint_initial_positions[path[i+1]])
        joint_dists.append(joint_dist)
        dist += joint_dist
        

    root2target_dist =  np.linalg.norm(temp_joint_positions[root_joint_index]-target_pose)
    #Check whether the target is within reach
    
    if root2target_dist > dist :
        # The target is unreachable
        for i in range(len(path) - 1):
            r = np.linalg.norm(target_pose - temp_joint_positions[path[i]])
            lamda = joint_dists[i] / r
            temp_joint_positions[path[i + 1]] = (1 - lamda)*temp_joint_positions[path[i]] + lamda*target_pose
    else :
        #The target is reachable
        b = joint_positions[root_joint_index]
        #Check whether the distance between the end effector
        difA = np.linalg.norm(temp_joint_positions[end_joint_index] - target_pose)
        while difA > 0.01:
            #STAGE 1: FORWARD REACHING
            temp_joint_positions[end_joint_index] = target_pose
            for i in range(len(path)-2,-1,-1):
                r = np.linalg.norm( temp_joint_positions[path[i+1]] - temp_joint_positions[path[i]])
                lamda = joint_dists[i] / r
                #Find the new joint positions pi
                temp_joint_positions[path[i]] = (1 - lamda)*temp_joint_positions[path[i+1]] + lamda * temp_joint_positions[path[i]]

            #STAGE 2: BACKWARD REACHING
            temp_joint_positions[root_joint_index] = b
            
            for i in range(len(path)-1):
                r = np.linalg.norm( temp_joint_positions[path[i+1]] - temp_joint_positions[path[i]])
                lamda = joint_dists[i] / r
                temp_joint_positions[path[i+1]] = (1 - lamda)*temp_joint_positions[path[i]] + lamda * temp_joint_positions[path[i+1]]
            
            difA = np.linalg.norm(temp_joint_positions[end_joint_index] - target_pose)
            
    
    #update orientation
    for i in range(len(path2)):
        #ignore root joint
        if i == 0 :
            continue
        
        dir = (temp_joint_positions[path2[i - 1]] - temp_joint_positions[path2[i]])/np.linalg.norm(temp_joint_positions[path2[i-1]] - temp_joint_positions[path2[i]])
        init_dir = (joint_initial_positions[path2[i - 1]] - joint_initial_positions[path2[i]])/np.linalg.norm(joint_initial_positions[path2[i-1]] - joint_initial_positions[path2[i]])
        rotation_angel = np.dot(dir,init_dir) / (np.linalg.norm(dir)*np.linalg.norm(init_dir))
        rotation_angel = -1.0 if rotation_angel < -1.0 + 0.00001 else rotation_angel
        rotation_angel = 1.0 if rotation_angel > 1.0 - 0.00001 else rotation_angel
        rotation_angel = np.arccos(rotation_angel) 
        rotation_vector = np.cross(init_dir,dir)
        normalized_vec =  rotation_vector/np.linalg.norm(rotation_vector)  if np.count_nonzero(rotation_vector) else np.array([0.,0.,0.])
        rotation = R.from_rotvec(rotation_angel * normalized_vec)
        orientation = rotation
        temp_joint_orientations[path2[i]] = orientation.as_quat()
        
    for i in range(len(path1)):
        #ignore end joint
        if i == 0 :
            continue
        dir = (temp_joint_positions[path1[i - 1]] - temp_joint_positions[path1[i]])/np.linalg.norm(temp_joint_positions[path1[i - 1]] - temp_joint_positions[path1[i]])
        init_dir = (joint_initial_positions[path1[i - 1]] - joint_initial_positions[path1[i]])/np.linalg.norm(joint_initial_positions[path1[i - 1]] - joint_initial_positions[path1[i]])
        rotation_angel = np.dot(dir,init_dir) / (np.linalg.norm(dir)*np.linalg.norm(init_dir))
        rotation_angel = -1.0 if rotation_angel < -1.0 + 0.00001 else rotation_angel
        rotation_angel = 1.0 if rotation_angel > 1.0 - 0.00001 else rotation_angel
        rotation_angel = np.arccos(rotation_angel) 
        rotation_vector = np.cross(init_dir,dir)
        normalized_vec =  rotation_vector/np.linalg.norm(rotation_vector)  if np.count_nonzero(rotation_vector) else np.array([0.,0.,0.])
        rotation = R.from_rotvec(rotation_angel * normalized_vec)
        orientation = rotation
        temp_joint_orientations[path1[i]] = orientation.as_quat()
    
    #update fullbody fk
    for index in range(len(joint_names)):
            #ignore root joint
            if index == 0:
                continue
            parent_orientation =R.from_quat(temp_joint_orientations[joint_parents[index]])
            parent_position = temp_joint_positions[joint_parents[index]]
            joint_offset = joint_initial_positions[index] - joint_initial_positions[joint_parents[index]]
            #joint_orientation =  parent_orientation 
            #temp_joint_orientations[index] =joint_orientation.as_quat()
            temp_joint_positions[index] =  parent_position + parent_orientation.apply(joint_offset)
    
    joint_positions = temp_joint_positions
    joint_orientations = temp_joint_orientations
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    path,path_name,path1,path2 = meta_data.get_path_from_root_to_end()
    joint_parents = meta_data.joint_parent
    joint_initial_positions = meta_data.joint_initial_position
    joint_names = meta_data.joint_name
    temp_joint_positions = joint_positions.copy()
    
    temp_joint_orientations = joint_orientations.copy()
    
    root_joint_index = joint_names.index(meta_data.root_joint)
    end_joint_index = joint_names.index(meta_data.end_joint)

    target_pose = np.array([joint_positions[root_joint_index][0] + relative_x,target_height,joint_positions[root_joint_index][2]+relative_z])
    joint_dists = []
    dist = 0
    
    for i in range(len(path)-1):
        joint_dist = np.linalg.norm(joint_initial_positions[path[i]] - joint_initial_positions[path[i+1]])
        joint_dists.append(joint_dist)
        dist += joint_dist
        

    root2target_dist =  np.linalg.norm(temp_joint_positions[root_joint_index]-target_pose)
    #Check whether the target is within reach
    
    if root2target_dist > dist :
        # The target is unreachable
        for i in range(len(path) - 1):
            r = np.linalg.norm(target_pose - temp_joint_positions[path[i]])
            lamda = joint_dists[i] / r
            temp_joint_positions[path[i + 1]] = (1 - lamda)*temp_joint_positions[path[i]] + lamda*target_pose
    else :
        #The target is reachable
        b = joint_positions[root_joint_index]
        #Check whether the distance between the end effector
        difA = np.linalg.norm(temp_joint_positions[end_joint_index] - target_pose)
        while difA > 0.001:
            #STAGE 1: FORWARD REACHING
            temp_joint_positions[end_joint_index] = target_pose
            for i in range(len(path)-2,-1,-1):
                r = np.linalg.norm( temp_joint_positions[path[i+1]] - temp_joint_positions[path[i]])
                lamda = joint_dists[i] / r
                #Find the new joint positions pi
                temp_joint_positions[path[i]] = (1 - lamda)*temp_joint_positions[path[i+1]] + lamda * temp_joint_positions[path[i]]

            #STAGE 2: BACKWARD REACHING
            temp_joint_positions[root_joint_index] = b
            
            for i in range(len(path)-1):
                r = np.linalg.norm( temp_joint_positions[path[i+1]] - temp_joint_positions[path[i]])
                lamda = joint_dists[i] / r
                temp_joint_positions[path[i+1]] = (1 - lamda)*temp_joint_positions[path[i]] + lamda * temp_joint_positions[path[i+1]]
            
            difA = np.linalg.norm(temp_joint_positions[end_joint_index] - target_pose)
            
    
    #update orientation
    for i in range(len(path2)):
        #ignore root joint
        if i == 0 :
            continue
        
        dir = (temp_joint_positions[path2[i - 1]] - temp_joint_positions[path2[i]])/np.linalg.norm(temp_joint_positions[path2[i-1]] - temp_joint_positions[path2[i]])
        init_dir = (joint_initial_positions[path2[i - 1]] - joint_initial_positions[path2[i]])/np.linalg.norm(joint_initial_positions[path2[i-1]] - joint_initial_positions[path2[i]])
        rotation_angel = np.dot(dir,init_dir) / (np.linalg.norm(dir)*np.linalg.norm(init_dir))
        rotation_angel = -1.0 if rotation_angel < -1.0 + 0.00001 else rotation_angel
        rotation_angel = 1.0 if rotation_angel > 1.0 - 0.00001 else rotation_angel
        rotation_angel = np.arccos(rotation_angel) 
        rotation_vector = np.cross(init_dir,dir)
        normalized_vec =  rotation_vector/np.linalg.norm(rotation_vector)  if np.count_nonzero(rotation_vector) else np.array([0.,0.,0.])
        rotation = R.from_rotvec(rotation_angel * normalized_vec)
        orientation = rotation
        temp_joint_orientations[path2[i]] = orientation.as_quat()
    
    path1.append(joint_parents[path1[-1]])
    for i in range(len(path1)):
        
        dir = (temp_joint_positions[path1[i - 1]] - temp_joint_positions[path1[i]])/np.linalg.norm(temp_joint_positions[path1[i - 1]] - temp_joint_positions[path1[i]])
        init_dir = (joint_initial_positions[path1[i - 1]] - joint_initial_positions[path1[i]])/np.linalg.norm(joint_initial_positions[path1[i - 1]] - joint_initial_positions[path1[i]])
        rotation_angel = np.dot(dir,init_dir) / (np.linalg.norm(dir)*np.linalg.norm(init_dir))
        rotation_angel = -1.0 if rotation_angel < -1.0 + 0.00001 else rotation_angel
        rotation_angel = 1.0 if rotation_angel > 1.0 - 0.00001 else rotation_angel
        rotation_angel = np.arccos(rotation_angel) 
        rotation_vector = np.cross(init_dir,dir)
        normalized_vec =  rotation_vector/np.linalg.norm(rotation_vector)  if np.count_nonzero(rotation_vector) else np.array([0.,0.,0.])
        rotation = R.from_rotvec(rotation_angel * normalized_vec)
        orientation = rotation
        temp_joint_orientations[path1[i]] = orientation.as_quat()

    
    joint_positions = temp_joint_positions
    joint_orientations = temp_joint_orientations
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations