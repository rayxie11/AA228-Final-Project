import numpy as np

def euler_rotation(psi, theta, phi):
    '''
    Compute the rotation matrix as Euler Angles ZYX
    Args:
        psi: rotation around z axis
        theta: rotation around y axis
        phi: rotation around x axis
    Return:
        R: rotation matrix
    '''
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    c_the = np.cos(theta)
    s_the = np.sin(theta)
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    R = np.array([[c_the*c_psi, s_phi*s_the*c_psi-c_phi*s_psi, c_phi*s_the*c_psi+s_phi*s_psi],
                  [c_the*s_psi, s_phi*s_the*s_psi+c_phi*c_psi, c_phi*s_the*s_psi-s_phi*c_psi],
                  [-s_the, s_phi*c_the, c_phi*c_the]])
    return R


class KinematicQuadcopter:
    '''
    In this quadcopter dynamic model, we are assuming a uniform mass quadcopter. The mass of the blade and motors
    are neglected. The length and width of the quadcopter is the same. The drag coefficient (b) of the quadcopter 
    is simplified to a long cylinder. The thrust coeffcient (kt) is 1.5e-5. The constant for calculating drag force
    (kd) is calculated with the cross-section of quadcopter as 0.01m^2. Motors 1 and 3 are on the roll axis. Motors
    2 and 4 are on the pitch axis.
    '''
    def __init__(self, pos, vel, mass, dim, thrust_max, thrust_min):
        self.t_max = thrust_max
        self.t_min = thrust_min
        self.input = np.zeros(4)
        self.b = 0.82                       # drag coefficient
        #self.kt = 1.5e-5                    # thrust coefficient
        self.kt = 1
        self.kd = 1.293*self.b*0.01/2       # kd
        self.mass = mass                    # mass
        self.dim = dim                      # dimension constant
        self.pos = np.array(pos[:3])        # position [x y z]
        self.ori = np.array(pos[3:])        # orientation row pitch yaw [psi theta phi]
        self.v = np.array(vel[:3])          # linear velocity [x' y' z']
        self.thetadot = np.array(vel[3:])   # row pitch yall derivatives [psi' theta' phi']
        self.calc_I()                   
        

    def calc_I(self):
        '''
        Compute the inertial matrix of quadcopter
        '''
        Ix = 1/12*self.mass*(self.dim**2+self.dim**2)
        Iz = 1/2*self.mass*(self.dim**2)
        self.I = np.array([[Ix, 0, 0],
                           [0, Ix, 0],
                           [0, 0, Iz]])
    
    def thrust(self, inputs):
        '''
        Compute the total thrust of the quadcopter
        Args:
            inputs: w_i^2 of each motor
        Return:
            T: total thrust
        '''
        T = np.array([0,0,self.kt*np.sum(inputs)])
        return T
    
    def torques(self, inputs):
        '''
        Compute the torques generated
        Args:
            inputs: w_i^2 of each motor
        Return:
            tau: torque vector
        '''
        tau = np.array([self.dim/2*self.kt*(inputs[0]-inputs[2]),
                        self.dim/2*self.kt*(inputs[1]-inputs[3]),
                        self.b*(inputs[0]-inputs[1]+inputs[2]-inputs[3])])
        return tau

    def acceleration(self, inputs, wind):
        '''
        Compute the linear acceleration of the quadcopter
        Args:
            inputs: w_i^2 of each motor
            wind: the wind force vector
        Return:
            a: linear acceleration
        '''
        g = np.array([0,0,-9.8])
        R = euler_rotation(self.ori[2],self.ori[1],self.ori[0])
        Tb = R@self.thrust(inputs)
        Fd = -self.kd*self.v
        a = g+1/self.mass*Tb+Fd+wind/self.mass
        return a
    
    def angular_acceleration(self, inputs, temp_w):
        '''
        Compute the angular acceleration of the quadcopter
        Args:
            inputs: w_i^2 of each motor
            temp_w: temporary angular velocity
        Return:
            angular_a: angular acceleration
        '''
        tau = self.torques(inputs)
        angular_a = np.linalg.inv(self.I)@(tau-np.cross(temp_w,self.I@temp_w))
        return angular_a
    
    def thetadot2omega(self):
        '''
        Convert derivatives of row, pitch, yaw to angular velocity
        Return:
            w: angular velocity
        '''
        psi = self.ori[0]
        theta = self.ori[1]
        W = np.array([[1, np.sin(psi)*np.tan(theta), np.cos(psi)*np.tan(theta)],
                      [0, np.cos(theta), -np.sin(psi)],
                      [0, np.sin(psi)/np.cos(theta), np.cos(psi)/np.cos(theta)]])
        w = np.linalg.inv(W)@self.thetadot
        return w
    
    def omega2thetadot(self, temp_w):
        '''
        Convert angular velocity to derivatives of row, pitch, yaw
        Args:
            temp_w: temporary angular velocity
        Return:
            thetadot: derivatives of row, pitch, yaw
        '''
        psi = self.ori[0]
        theta = self.ori[1]
        W = np.array([[1, np.sin(psi)*np.tan(theta), np.cos(psi)*np.tan(theta)],
                      [0, np.cos(theta), -np.sin(psi)],
                      [0, np.sin(psi)/np.cos(theta), np.cos(psi)/np.cos(theta)]])
        thetadot = W@temp_w
        return thetadot

    def check_valid_update(self, pos, ori, env):
        '''
        Check whether the updated state is valid: no flipping and position within bounds of environment
        not colliding with obstacles
        Args:
            pos: updated position
            ori: updated orientation
            env: environment quadcopter is in
        Return:
            True/False
        '''
        if ori[0] >= np.pi or ori[1] >= np.pi or ori[2] >= np.pi:
            return False
        if not env.check_state_in_bound(pos) or env.check_state_in_obstacle(pos):
            return False
        return True

    def update(self, dt, env):
        '''
        Update quadcopter state
        Args:
            dt: time step
            env: environment quadcopter is in
        '''
        #inputs = np.ones(4)*100
        inputs = np.array([10,10,5,5])
        #inputs = np.zeros(4)
        temp_w = self.thetadot2omega()
        a = self.acceleration(inputs)
        omegadot = self.angular_acceleration(inputs, temp_w)
        temp_w += omegadot*dt
        temp_thetadot = self.omega2thetadot(temp_w)
        temp_ori = self.ori+temp_thetadot*dt
        temp_v = self.v+a*dt
        temp_pos = self.pos+temp_v*dt
        if self.check_valid_update(temp_pos, temp_ori, env):
            self.pos = temp_pos
            self.ori = temp_ori
            self.thetadot = temp_thetadot
            self.v = temp_v
            return True
        else:
            return False
        '''
        self.thetadot2omega()
        a = self.acceleration(inputs)
        omegadot = self.angular_acceleration(inputs)
        self.w += omegadot*dt
        self.omega2thetadot()
        self.ori += self.thetadot*dt
        self.v += a*dt
        self.pos += self.v*dt
        '''

        
