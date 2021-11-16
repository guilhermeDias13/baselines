import numpy as np

class MirrorFunctions:
    def __init_(self, pos_ini_state=0, num_of_joints=20):
        self.pos_ini_state = pos_ini_state
        self.num_of_joints = num_of_joints

    def mirror_state(self, state):
        mirrored_state = np.copy(state)

        # Joints
        state[self.pos_ini_state:self.pos_ini_state + self.num_of_joints ] = \
            self.mirror_array_of_joints(state[self.pos_ini_state:self.pos_ini_state + self.num_of_joints - 1])

        # Joints Derivative
        state[self.pos_ini_state + self.num_of_joints: self.pos_ini_state + 2 * self.num_of_joints] = \
            self.mirror_array_of_joints(state[self.pos_ini_state + self.num_of_joints:
                                              self.pos_ini_state + 2 * self.num_of_joints])

        # Orientation Angle
        mirrored_state[41] = -state[41]
        mirrored_state[42] = -state[42]
        state[41] = mirrored_state[41]
        state[41] = mirrored_state[42]
        
        # Center of Mass in Y
        mirrored_state[46] = -state[46]
        mirrored_state[49] = -state[49]
        state[46] = mirrored_state[46]
        state[49] = mirrored_state[49]

        # Torso Angular Velocity in Z
        mirrored_state[53] = -state[53]
        mirrored_state[56] = -state[56]
        state[53] = mirrored_state[53]
        state[56] = mirrored_state[56]

        # Torso Acceleration in Y
        mirrored_state[58] = -state[58]
        mirrored_state[61] = -state[61]
        state[58] = mirrored_state[58]
        state[61] = mirrored_state[61]

        # Foot Force Resistance Coordinates in X
        mirrored_state[63] = state[75]
        mirrored_state[75] = state[63]
        state[63] = mirrored_state[75]
        state[75] = mirrored_state[63]

        # Foot Force Resistance Coordinates Derivatives in X
        mirrored_state[66] = state[78]
        mirrored_state[78] = state[66]
        state[66] = mirrored_state[78]
        state[78] = mirrored_state[66]

        # Foot Force Resistance Coordinates in Y
        mirrored_state[64] = -state[76]
        mirrored_state[76] = -state[64]
        state[64] = mirrored_state[76]
        state[76] = mirrored_state[64]

        # Foot Force Resistance Coordinates Derivatives in Y
        mirrored_state[64] = -state[76]
        mirrored_state[76] = -state[64]
        state[64] = mirrored_state[76]
        state[76] = mirrored_state[64]

        # Foot Force Resistance Coordinates in Z
        mirrored_state[65] = state[77]
        mirrored_state[77] = state[65]
        state[65] = mirrored_state[77]
        state[77] = mirrored_state[65]
        
        # Foot Force Resistance Coordinates Derivatives in Z
        mirrored_state[68] = state[80]
        mirrored_state[80] = state[68]
        state[68] = mirrored_state[80]
        state[80] = mirrored_state[68]

        # Foot Force Resistance in X
        mirrored_state[69] = state[81]
        mirrored_state[81] = state[69]
        state[69] = mirrored_state[81]
        state[81] = mirrored_state[69]
        
        # Foot Force Resistance Derivatives in X
        mirrored_state[72] = state[84]
        mirrored_state[84] = state[72]
        state[72] = mirrored_state[84]
        state[84] = mirrored_state[72]

        # Foot Force Resistance in Y
        mirrored_state[70] = -state[82]
        mirrored_state[82] = -state[70]
        state[70] = mirrored_state[82]
        state[82] = mirrored_state[70]

        # Foot Force Resistance Derivatives in Y
        mirrored_state[73] = -state[85]
        mirrored_state[85] = -state[73]
        state[73] = mirrored_state[85]
        state[85] = mirrored_state[73]

        # Foot Force Resistance in Z
        mirrored_state[71] = state[83]
        mirrored_state[83] = state[71]
        state[71] = mirrored_state[83]
        state[83] = mirrored_state[71]

        # Foot Force Resistance Derivatives in Z
        mirrored_state[74] = state[86]
        mirrored_state[86] = state[74]
        state[74] = mirrored_state[86]
        state[86] = mirrored_state[74]

        # Foot Counter
        mirrored_state[87] = state[88]
        mirrored_state[88] = state[87]
        state[87] = mirrored_state[88]
        state[88] = mirrored_state[87]

        return state


    def mirror_array_of_joints(self, state):
        mirrored_array = np.copy(state)

        # ShoulderPitch
        mirrored_array[0] = state[4]
        mirrored_array[4] = state[0]

        # ShoulderYaw
        mirrored_array[1] = -state[5]
        mirrored_array[5] = -state[1]

        # ArmRoll
        mirrored_array[2] = -state[6]
        mirrored_array[6] = -state[2]

        # ArmYaw
        mirrored_array[3] = -state[7]
        mirrored_array[7] = -state[3]

        # HipYawPitch
        mirrored_array[8] = -state[8]
        mirrored_array[14] = -state[14]

        # HipRoll
        mirrored_array[9] = -state[15]
        mirrored_array[15] = -state[9]

        # HipPitch
        mirrored_array[10] = state[16]
        mirrored_array[16] = state[10]

        # KneePitch
        mirrored_array[11] = state[17]
        mirrored_array[17] = state[11]

        # FootPitch
        mirrored_array[12] = state[18]
        mirrored_array[18] = state[12]

        # FootRoll
        mirrored_array[13] = -state[19]
        mirrored_array[19] = -state[13]

        return mirrored_array