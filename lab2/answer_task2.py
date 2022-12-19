# 以下部分均为可更改部分

from answer_task1 import *
from smooth_utils import *
from scipy.spatial import KDTree
class CharacterController():
    def __init__(self, controller) -> None:
        self.motions = []
        self.motions.append(BVHMotion('motion_material/walk_forward.bvh'))
        self.motions.append(BVHMotion('motion_material/walk_and_turn_left.bvh'))
        self.motions.append(BVHMotion('motion_material/walk_and_ture_right.bvh'))
        self.motions.append(BVHMotion('motion_material/run_forward.bvh'))
        self.motions.append(BVHMotion('motion_material/idle.bvh'))
        pose = []
        for i in range(len(self.motions)):
            pose.append(Pose(self.motions[i]))
        self.pose = pose
        self.cur_pose = self.pose[4]
        self.cur_motion = self.cur_pose.motion
        self.joint_name = self.cur_motion.joint_name
        self.controller = controller
        self.cur_root_pos = self.cur_pose.pos_list[0]
        self.cur_root_rot = self.cur_pose.rot_list[0]
        self.cur_frame = 0
        self.cur_lToe_pos = self.cur_pose.lToeJoint_pos_list[1] 
        self.cur_rToe_pos = self.cur_pose.rToeJoint_pos_list[1] 
        self.cur_lToe_vel = self.cur_pose.lToeJoint_vel_list[1]
        self.cur_rToe_vel = self.cur_pose.rToeJoint_vel_list[1]
        pass
    
    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list,
                     current_gait
                     ):
        '''
        此接口会被用于获取新的期望状态
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态,以及一个额外输入的步态
        简单起见你可以先忽略步态输入,它是用来控制走路还是跑步的
            desired_pos_list: 期望位置, 6x3的矩阵, 每一行对应0，20，40...帧的期望位置(水平)， 期望位置可以用来拟合根节点位置也可以是质心位置或其他
            desired_rot_list: 期望旋转, 6x4的矩阵, 每一行对应0，20，40...帧的期望旋转(水平), 期望旋转可以用来拟合根节点旋转也可以是其他
            desired_vel_list: 期望速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望速度(水平), 期望速度可以用来拟合根节点速度也可以是其他
            desired_avel_list: 期望角速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望角速度(水平), 期望角速度可以用来拟合根节点角速度也可以是其他
        
        Output: 同作业一,输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            输出三者顺序需要对应
            controller 本身有一个move_speed属性,是形状(3,)的ndarray,
            分别对应着面朝向移动速度,侧向移动速度和向后移动速度.目前根据LAFAN的统计数据设为(1.75,1.5,1.25)
            如果和你的角色动作速度对不上,你可以在init或这里对属性进行修改
        '''
        
        
        pos_list = desired_pos_list.copy()
        rot_list = desired_rot_list.copy()
        pos_list = desired_pos_list - desired_pos_list[0]
        rot = R.from_quat(desired_rot_list[2])
        pos_list = rot.inv().apply(pos_list)
        for r in range(len(rot_list)):
            rot_list[r] = (R.from_quat(rot_list[r])*rot.inv()).as_quat()
        
        #旋转和位移到局部坐标系
        target_facing_dir = rot.apply(np.array([0,0,1]))
        target_facing_dir = [target_facing_dir[0],target_facing_dir[2]]
        
        
        best_pose = self.pose[0]
        best_cost = 10000
        best_frame = 0
        for i in range(len(self.pose)):
            for frame in range(self.pose[i].motion_length):
                this_cost = self.pose[i].compute_cost(pos_list, rot_list, desired_vel_list, desired_avel_list,self,frame)
                if this_cost < best_cost:
                    best_cost = this_cost
                    best_pose = self.pose[i]
                    best_frame = frame
                    
        motion = best_pose.get_motion()
        self.cur_pose = best_pose
        self.cur_frame = best_frame
        joint_name = motion.joint_name
       
        #旋转和位移到全局
        motion = motion.translation_and_rotation(best_frame,desired_pos_list[0,[0,2]],target_facing_dir)
        
        #更新当前pose状态
        joint_translation, joint_orientation = motion.batch_forward_kinematics()
        joint_translation = joint_translation[self.cur_frame]
        joint_orientation = joint_orientation[self.cur_frame]
        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]
        self.cur_frame = (self.cur_frame + 1) % motion.motion_length
        self.cur_lToe_pos = self.cur_pose.lToeJoint_pos_list[self.cur_frame]
        self.cur_rToe_pos = self.cur_pose.rToeJoint_pos_list[self.cur_frame]
        self.cur_lToe_vel = self.cur_pose.lToeJoint_vel_list[self.cur_frame]
        self.cur_rToe_vel = self.cur_pose.rToeJoint_vel_list[self.cur_frame]
        '''
        
        
        # 一个简单的例子，输出第i帧的状态
        cur_motion = self.pose[3].get_motion()
        joint_name = cur_motion.joint_name
        joint_translation, joint_orientation = cur_motion.batch_forward_kinematics()
        joint_translation = joint_translation[self.cur_frame]
        joint_orientation = joint_orientation[self.cur_frame]
        
        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]
        self.cur_frame = (self.cur_frame + 1) % cur_motion.motion_length
        '''
        '''
        cur_motion = self.motions[1]
        joint_name = cur_motion.joint_name
        joint_translation, joint_orientation = cur_motion.batch_forward_kinematics()
        joint_translation = joint_translation[self.cur_frame]
        joint_orientation = joint_orientation[self.cur_frame]
        
        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]
        self.cur_frame = (self.cur_frame + 1) % cur_motion.motion_length
        '''
        return joint_name, joint_translation, joint_orientation
    
    
    def sync_controller_and_character(self, controller, character_state):
        '''
        这一部分用于同步你的角色和手柄的状态
        更新后很有可能会出现手柄和角色的位置不一致，这里可以用于修正
        让手柄位置服从你的角色? 让角色位置服从手柄? 或者插值折中一下?
        需要你进行取舍
        Input: 手柄对象，角色状态
        手柄对象我们提供了set_pos和set_rot接口,输入分别是3维向量和四元数,会提取水平分量来设置手柄的位置和旋转
        角色状态实际上是一个tuple, (joint_name, joint_translation, joint_orientation),为你在update_state中返回的三个值
        你可以更新他们,并返回一个新的角色状态
        '''
        
        # 一个简单的例子，将手柄的位置与角色对齐
        #controller.set_pos(self.cur_root_pos)
        #controller.set_rot(self.cur_root_rot)
        
        #self.cur_root_pos = controller.get_pos()
        return character_state
    # 你的其他代码,state matchine, motion matching, learning, etc.
    



class Pose:
    def __init__(self, motion):
        self.motion = motion
        self.motion_length = motion.motion_length
        self.joint_name = motion.joint_name
        self.frame = 0
        self.root_height = 0
        self.pos_list,self.rot_list,self.vel_list,self.avel_list,self.rToeJoint_pos_list,self.lToeJoint_pos_list,self.rToeJoint_vel_list,self.lToeJoint_vel_list = self.prepare(motion)

    def prepare(self, motion):
        '''
        用于对motion进行预处理
        '''
        self.motion = self.motion.translation_and_rotation(0,np.array([0,0]),np.array([0,1]))
        joint_name = self.motion.joint_name
        joint_translation , joint_orientation = self.motion.batch_forward_kinematics()
        rToeJoint_pos_list = joint_translation[:,joint_name.index('rToeJoint'),:]
        lToeJoint_pos_list = joint_translation[:,joint_name.index('lToeJoint'),:]
        rToeJoint_vel_list = np.zeros_like(rToeJoint_pos_list)
        lToeJoint_vel_list = np.zeros_like(lToeJoint_pos_list)
        
        pos_list = self.motion.joint_position[:, 0, :].copy()
        pos_list[:, 1].fill(0)
        self.root_height = self.motion.joint_position[:, 0, :] - pos_list
        vel_list = np.zeros_like(pos_list)
        rot_list = motion.joint_rotation[:, 0, :]
        avel_list = np.zeros_like(rot_list)
        for i in range(1, motion.motion_length):
            vel_list[i] = (pos_list[i] - pos_list[i - 1]) * 60
            rToeJoint_vel_list[i] = (rToeJoint_pos_list[i] - rToeJoint_pos_list[i - 1]) * 60
            lToeJoint_vel_list[i] = (lToeJoint_pos_list[i] - lToeJoint_pos_list[i - 1]) * 60
        
        avel_list = quat_to_avel(rot_list, 1/60)
        vel_list = vel_list[1:]
        rToeJoint_pos_list = rToeJoint_vel_list - pos_list
        lToeJoint_pos_list = lToeJoint_vel_list - pos_list
        return pos_list,rot_list,vel_list,avel_list,rToeJoint_pos_list,lToeJoint_pos_list,rToeJoint_vel_list,lToeJoint_vel_list
    
    def get(self, frame):
        '''
        用于获取某一帧的特征
        '''
        frame_ = frame % self.avel_list.shape[0]
        return self.pos_list[frame_], self.rot_list[frame_], self.vel_list[frame_], self.avel_list[frame_]
    
    def get_motion(self):
        return self.motion
    
    
    def compute_cost(self, 
                    desired_pos_list,
                    desired_rot_list,
                    desired_vel_list,
                    desired_avel_list,
                    characterController,
                    frame
                     ):
        '''
        用于计算期望的特征和当前特征的差异
        '''
        weight = 2
        #frame_list = [0,20]
        cost = 0
        #for i in range(len(frame_list)):
        #future_frame = frame_list[i]
        pos, rot, vel, avel = self.get(frame)
        future_pos ,future_rot, future_vel, future_avel = self.get(frame + 20)
        
        desired_pos = desired_pos_list[0]
        desired_rot = desired_rot_list[0]
        desired_vel = desired_vel_list[0]
        desired_avel = desired_avel_list[0]
        
        rotation = R.from_quat(rot)
        #20 frame后的期望
        future_desired_pos = rotation.inv().apply(desired_pos_list[1])
        future_desired_rot =  rotation.inv() * R.from_quat(desired_rot_list[1])
        future_desired_vel = desired_vel_list[1]
        future_desired_avel = desired_avel_list[1]
        
        pos_cost = np.linalg.norm(pos - desired_pos)
        rot_cost = (R.from_quat(rot) * R.from_quat(desired_rot).inv()).as_euler('xyz')
        rot_cost = np.linalg.norm(rot_cost)
        vel_cost = np.linalg.norm(vel - desired_vel)
        avel_cost = np.linalg.norm(avel - desired_avel)
        
        future_pos_cost = np.linalg.norm(future_pos - future_desired_pos)
        future_rot_cost = (R.from_quat(future_rot) * future_desired_rot.inv()).as_euler('xyz')
        future_rot_cost = np.linalg.norm(future_rot_cost)
        future_vel_cost = np.linalg.norm(future_vel - future_desired_vel)
        future_avel_cost = np.linalg.norm(future_avel - future_desired_avel)
        
       # weight *= 1.2
        #if i == 0:
           # cost = (pos_cost + rot_cost + vel_cost + avel_cost)*weight
        #else:
        cost += 2*pos_cost + 2*rot_cost + 2*vel_cost + 2*avel_cost + future_pos_cost + future_rot_cost + future_vel_cost + future_avel_cost

        cur_lToe_pos = characterController.cur_lToe_pos
        cur_rToe_pos = characterController.cur_rToe_pos
        cur_lToe_vel = characterController.cur_lToe_vel
        cur_rToe_vel = characterController.cur_rToe_vel
        lToe_pos_cost = self.lToeJoint_pos_list[frame] - cur_lToe_pos
        lToe_pos_cost = np.linalg.norm(lToe_pos_cost)
        rToe_pos_cost = self.rToeJoint_pos_list[frame] - cur_rToe_pos
        rToe_pos_cost = np.linalg.norm(rToe_pos_cost)
        lToe_vel_cost = self.lToeJoint_vel_list[frame] - cur_lToe_vel
        lToe_vel_cost = np.linalg.norm(lToe_vel_cost)
        rToe_vel_cost = self.rToeJoint_vel_list[frame] - cur_rToe_vel
        rToe_vel_cost = np.linalg.norm(rToe_vel_cost)
         
        cost += 5 * lToe_pos_cost + 5 * rToe_pos_cost + 2 * lToe_vel_cost + 5 * rToe_vel_cost
        return cost
    
