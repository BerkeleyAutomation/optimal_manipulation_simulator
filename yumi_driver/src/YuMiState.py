'''
State Encapsulation for YuMi robot
Author: Jacky
'''
class YuMiState:

    NUM_STATES = 7
    NAME = "YuMi"

    def __init__(self, vals = [None] * NUM_STATES, gripper_value = 0):
        self.raw_state = vals[::]
        self.gripper_value = gripper_value

    def __str__(self):
        return str(self.raw_state)
            
    def __repr__(self):
        return "YuMiState({0})".format(self.raw_state)
        
    @property
    def joint_1(self):
        return self.state[0]
        
    def set_joint_1(self, val):
        self.state[0] = val
        return self
        
    @property
    def joint_2(self):
        return self.state[1]
        
    def set_joint_2(self, val):
        self.state[1] = val
        return self
        
    @property
    def joint_3(self):
        return self.state[2]
        
    def set_joint_3(self, val):
        self.state[2] = val
        return self
        
    @property
    def joint_4(self):
        return self.state[3]
        
    def set_joint_4(self, val):
        self.state[3] = val
        return self
        
    @property
    def joint_5(self):
        return self.state[4]
        
    def set_joint_5(self, val):
        self.state[4] = val
        return self
        
    @property
    def joint_6(self):
        return self.state[5]
        
    def set_joint_6(self, val):
        self.state[5] = val
        return self
        
    @property
    def joint_7(self):
        return self.state[6]
        
    def set_joint_7(self, val):
        self.state[6] = val
        return self

    @property
    def gripper_value(self):
        return self.gripper_value
    
    def set_gripper_value(self, val):
        self.gripper_value = val
        return self
        
    def copy(self):
        return YuMiState(self.raw_state)
