import numpy as np

class MirrorFunctions:
    def _init_(self, pos_ini_state=0, num_of_joints=20):
        self.pos_ini_state = pos_ini_state
        self.num_of_joints = num_of_joints

    def mirror_state(self, state):
        mirrored_state = np.copy(state)

        # Joints
        mirrored_state[self.pos_ini_state:self.pos_ini_state + self.num_of_joints - 1] = \
            self.mirror_array_of_joints(state[self.pos_ini_state:self.pos_ini_state + self.num_of_joints - 1], self.pos_ini_state)

        # Joints Derivative
        mirrored_state[self.pos_ini_state + self.num_of_joints - 1: self.pos_ini_state + 2 * self.num_of_joints - 1] = \
            self.mirror_array_of_joints(state[self.pos_ini_state + self.num_of_joints - 1:
                                              self.pos_ini_state + 2 * self.num_of_joints - 1], self.pos_ini_state + self.num_of_joints)

        # Orientation Angle
        mirrored_state[41] = -state[41]
        mirrored_state[42] = -state[42]

        # Center of Mass in Y
        mirrored_state[46] = -state[46]
        mirrored_state[49] = -state[49]

        # Torso Angular Velocity in Z
        mirrored_state[53] = -state[53]
        mirrored_state[56] = -state[56]

        # Torso Acceleration in Y
        mirrored_state[58] = -state[58]
        mirrored_state[61] = -state[61]

        # Foot Force Resistance Coordinates in X
        mirrored_state[63] = state[75]
        mirrored_state[75] = state[63]

        # Foot Force Resistance Coordinates Derivatives in X
        mirrored_state[66] = state[78]
        mirrored_state[78] = state[66]

        # Foot Force Resistance Coordinates in Y
        mirrored_state[64] = -state[76]
        mirrored_state[76] = -state[64]

        # Foot Force Resistance Coordinates Derivatives in Y
        mirrored_state[64] = -state[76]
        mirrored_state[76] = -state[64]

        # Foot Force Resistance Coordinates in Z
        mirrored_state[65] = state[77]
        mirrored_state[77] = state[65]

        # Foot Force Resistance Coordinates Derivatives in Z
        mirrored_state[68] = state[80]
        mirrored_state[80] = state[68]

        # Foot Force Resistance in X
        mirrored_state[69] = state[81]
        mirrored_state[81] = state[69]

        # Foot Force Resistance Derivatives in X
        mirrored_state[72] = state[84]
        mirrored_state[84] = state[72]

        # Foot Force Resistance in Y
        mirrored_state[70] = -state[82]
        mirrored_state[82] = -state[70]

        # Foot Force Resistance Derivatives in Y
        mirrored_state[73] = -state[85]
        mirrored_state[85] = -state[73]

        # Foot Force Resistance in Z
        mirrored_state[71] = state[83]
        mirrored_state[83] = state[71]

        # Foot Force Resistance Derivatives in Z
        mirrored_state[74] = state[86]
        mirrored_state[86] = state[74]

        # Foot Counter
        mirrored_state[87] = state[88]
        mirrored_state[88] = state[87]


    def mirror_array_of_joints(self, state, pos_ini_state):
        mirrored_array = np.copy(state)

        # ShoulderPitch
        mirrored_array[pos_ini_state] = state[pos_ini_state + 4]
        mirrored_array[pos_ini_state + 4] = state[pos_ini_state]

        # ShoulderYaw
        mirrored_array[pos_ini_state + 1] = -state[pos_ini_state + 5]
        mirrored_array[pos_ini_state + 5] = -state[pos_ini_state + 1]

        # ArmRoll
        mirrored_array[pos_ini_state + 2] = -state[pos_ini_state + 6]
        mirrored_array[pos_ini_state + 6] = -state[pos_ini_state + 2]

        # ArmYaw
        mirrored_array[pos_ini_state + 3] = -state[pos_ini_state + 7]
        mirrored_array[pos_ini_state + 7] = -state[pos_ini_state + 1]

        # HipYawPitch
        mirrored_array[pos_ini_state + 8] = state[pos_ini_state + 8]
        mirrored_array[pos_ini_state + 14] = state[pos_ini_state + 14]

        # HipRoll
        mirrored_array[pos_ini_state + 9] = -state[pos_ini_state + 15]
        mirrored_array[pos_ini_state + 15] = -state[pos_ini_state + 9]

        # HipPitch
        mirrored_array[pos_ini_state + 10] = state[pos_ini_state + 16]
        mirrored_array[pos_ini_state + 16] = state[pos_ini_state + 10]

        # KneePitch
        mirrored_array[pos_ini_state + 11] = state[pos_ini_state + 17]
        mirrored_array[pos_ini_state + 17] = state[pos_ini_state + 11]

        # FootPitch
        mirrored_array[pos_ini_state + 12] = state[pos_ini_state + 18]
        mirrored_array[pos_ini_state + 18] = state[pos_ini_state + 12]

        # FootRoll
        mirrored_array[pos_ini_state + 13] = -state[pos_ini_state + 19]
        mirrored_array[pos_ini_state + 19] = -state[pos_ini_state + 13]

        return mirrored_array