import numpy as np
from numpy import linalg as LA
import random

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.075
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = 2.0
        self.accel = 7.5
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of agent entities, リーダーを表している
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = True
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # define front or not
        self.front = False

# Entityクラスを継承したFollwerクラスを作成
class Follower(Entity):
    def __init__(self):
        super(Follower, self).__init__()
        # フォロワの基準距離
        self.r_F = {"r1": 0.2, "r2": 0.3, "r3": 0.45, "r4": 0.6, "r5": 0.8}
        # リーダーの基準距離
        self.r_L = {"r1d": 0.2, "r2d":  0.5, "r5d": 1.0}
        # 微小な値
        self.delta = 10e-2
        # フォロワ間作用の係数
        self.k_FF_coh = 2.25; self.k_FF_col = 3; self.k_FF_bar = 100;
        # リーダーフォロワ間の係数
        self.k_FL_col = 2; self.k_FL_bar = 100;   
        # フォロワ-障害物間作用の係数
        self.k_Fland_coh = 0; self.k_Fland_col = 3; self.k_Fland_bar = 100;
        # フォロワのrendering時の色
        self.color = np.array([0, 0, 1])
        
    def __calc_vec_FF(self, followers):
        # フォロワiに対するフォロワjと障害物の相対ベクトルの取得
        vec_FFs = []
        for follower in followers:
            if follower == self: continue
            vec_FF = follower.state.p_pos - self.state.p_pos
            vec_FFs.append(vec_FF)
        
        return vec_FFs
    
    def __calc_vec_FOs(self, obstacles):
        # フォロワiに対する障害物の相対ベクトルの取得
        vec_FOs = []
        for O in obstacles:
            # 障害物は表面から力を受けるようにする
            norm = LA.norm(O.state.p_pos - self.state.p_pos)
            vec_F_O = (O.state.p_pos - self.state.p_pos) / norm * (norm - O.size)
            vec_FOs.append(vec_F_O)
        
        return vec_FOs
    
    def __calc_vec_FL(self, leaders):
        # フォロワiに対するリーダーjの相対ベクトルの取得
        vec_FLs = []
        for leader in leaders:
            vec_FL = leader.state.p_pos - self.state.p_pos
            vec_FLs.append(vec_FL) 
            
        return vec_FLs
    
    def __calc_vel_FF(self, vec_FFs: list, vec_Flands: list):
        final_vel_FF = np.array([0., 0.])
        for vec_FF in vec_FFs:
            dist = LA.norm(vec_FF)
            if dist <= self.r_F["r1"]: vel_FF = np.array([0., 0.])
            elif dist <= self.r_F["r2"]:
                vel_FF = (- self.k_FF_col - np.log((self.r_F["r3"] - self.r_F["r1"]) / (dist - self.r_F["r1"])) + self.k_FF_bar * ((dist - self.r_F["r2"]) / (dist - self.r_F["r1"]))) * (vec_FF / dist)   
            elif dist <= self.r_F["r3"]:
                vel_FF = (- self.k_FF_col - np.log((self.r_F["r3"] - self.r_F["r1"]) / (dist - self.r_F["r1"]))) * (vec_FF / dist)
            elif dist <= self.r_F["r4"] - self.delta:
                vel_FF = - self.k_FF_col * (vec_FF / dist)
            elif dist <= self.r_F["r4"]:
                vel_FF = self.k_FF_col * (vec_FF / dist) * ((dist-self.r_F["r4"]) / self.delta)
            elif dist <= self.r_F["r4"] + self.delta: 
                vel_FF = self.k_FF_coh * (vec_FF / dist) * ((dist-self.r_F["r4"]) / self.delta)
            elif dist <= self.r_F["r5"] - self.delta:
                vel_FF = self.k_FF_coh * (vec_FF / dist)
            elif dist <= self.r_F["r5"]: 
                vel_FF = - self.k_FF_coh * (vec_FF / dist) * ((dist-self.r_F["r5"]) / self.delta)
            else: vel_FF = np.array([0., 0.])
            
            final_vel_FF += vel_FF
        
        for vec_Fland in vec_Flands:
            dist = LA.norm(vec_Fland)
            if dist <= self.r_F["r1"]: vel_Fland = np.array([0., 0.])
            elif dist <= self.r_F["r2"]:
                vel_Fland = (- self.k_Fland_col - np.log((self.r_F["r3"] - self.r_F["r1"]) / (dist - self.r_F["r1"])) + self.k_Fland_bar * ((dist - self.r_F["r2"]) / (dist - self.r_F["r1"]))) * (vec_Fland / dist)   
            elif dist <= self.r_F["r3"]:
                vel_Fland = (- self.k_Fland_col - np.log((self.r_F["r3"] - self.r_F["r1"]) / (dist - self.r_F["r1"]))) * (vec_Fland / dist)
            elif dist <= self.r_F["r4"] - self.delta:
                vel_Fland = - self.k_Fland_col * (vec_Fland / dist)
            elif dist <= self.r_F["r4"]:
                vel_Fland = self.k_Fland_col * (vec_Fland / dist) * ((dist-self.r_F["r4"]) / self.delta)
            elif dist <= self.r_F["r4"] + self.delta: 
                vel_Fland = self.k_Fland_coh * (vec_Fland / dist) * ((dist-self.r_F["r4"]) / self.delta)
            elif dist <= self.r_F["r5"] - self.delta:
                vel_Fland = self.k_Fland_coh * (vec_Fland / dist)
            elif dist <= self.r_F["r5"]: 
                vel_Fland = - self.k_Fland_coh * (vec_Fland / dist) * ((dist-self.r_F["r5"]) / self.delta)
            else: vel_Fland = np.array([0., 0.])
            
            final_vel_FF += vel_Fland 
        
        return final_vel_FF
    
    def __calc_vel_FL(self, vec_FLs: list):
        final_vel_FL = np.array([0., 0.])
        for vec_FL in vec_FLs:
            dist = LA.norm(vec_FL)
            if dist <= self.r_L["r1d"]: vel_FL = np.array([0, 0])
            elif dist <= self.r_L["r2d"]: 
                vel_FL = self.k_FL_bar * (vec_FL / dist) * (dist - self.r_L["r2d"]) / (dist - self.r_L["r1d"])\
                        - self.k_FL_col * (vec_FL / dist)
            elif dist <= self.r_L["r5d"] - self.delta:
                vel_FL = - self.k_FL_col * (vec_FL / dist)
            elif dist <= self.r_L["r5d"]: 
                vel_FL = self.k_FL_col * (vec_FL / dist) * ((dist-self.r_L["r5d"]) / self.delta)
            else: vel_FL = np.array([0., 0.])
            
            final_vel_FL += vel_FL 
        
        return final_vel_FL   
    
    def calc_follower_input(self, agents, followers, obstacles):
        vec_FFs = self.__calc_vec_FF(followers)
        vec_Flands = self.__calc_vec_FOs(obstacles)
        vec_FLs = self.__calc_vec_FL(agents)
        final_vel_FF = self.__calc_vel_FF(vec_FFs, vec_Flands)
        final_vel_FL = self.__calc_vel_FL(vec_FLs)
        # followerの入力
        self.state.p_vel =  final_vel_FF + final_vel_FL
        
# properties of obstacles entities
class Obstacle(Entity):
    def __init__(self):
        super(Obstacle, self).__init__()
        self.color = np.array([0, 0.5, 0])
        self.have_vel = False
        self.max_speed = 0.5
        self.init_pos = np.array([0, 0])
        self.max_range = 1
        self.have_goal = False
        self.goal = np.array([0, 0])
    def set_vel(self):
        if not self.have_goal:
            x_sign = -1. if np.random.rand() < 0.01 else 1.
            y_sign = -1. if np.random.rand() < 0.01 else 1.
            if LA.norm(self.state.p_pos - self.init_pos) >= self.max_range:
                x_sign = -1.
                y_sign = -1.
            self.state.p_vel[0] *= x_sign
            self.state.p_vel[1] *= y_sign
            self.state.p_vel = self.state.p_vel.clip(-self.max_speed, self.max_speed)
        else:
            unit_vec = (self.goal - self.state.p_pos) / LA.norm(self.goal - self.state.p_pos)
            self.state.p_vel = 0.4 * unit_vec
        
# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        # leader
        self.agents = []
        # follower
        self.followers = []
        # obstacles
        self.obstacles = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep -> 25FPS
        self.dt = 0.035
        # physical damping -> バネ、ダンパーのダンパー,速度の減衰具合を表す
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # record episode num
        self.num_episodes = 0
        # for rendering
        self.box = []

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.followers + self.obstacles

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # gather agent action forces
    def __apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force

     # get collision forces for any contact between two entities
    def __get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        if dist < 1e-6: dist = 1e-6
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        # if force_a is not None and force_b is not None and force_a[0] >= 1e-5: 
        #     print(force_a)
        return [force_a, force_b]
    
    # gather physical forces acting on entities
    def __apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.__get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    def __integrate_state(self, p_force):
        # leaderの位置を更新
        for i, leader in enumerate(self.agents):
            if not leader.movable: continue
            leader.state.p_vel = leader.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                leader.state.p_vel += (p_force[i] / leader.mass) * self.dt
            if leader.max_speed is not None:
                speed = np.sqrt(np.square(leader.state.p_vel[0]) + np.square(leader.state.p_vel[1]))
                if speed > leader.max_speed:
                    leader.state.p_vel = leader.state.p_vel / np.sqrt(np.square(leader.state.p_vel[0]) +
                                                                    np.square(leader.state.p_vel[1])) * leader.max_speed
            leader.state.p_pos += leader.state.p_vel * self.dt
        # followerの位置を更新
        for i, follower in enumerate(self.followers):
            follower.calc_follower_input(self.agents, self.followers, self.obstacles)
            follower.state.p_pos += follower.state.p_vel * self.dt
        # obstacleの位置を更新
        for i, obstacle in enumerate(self.obstacles):
            if obstacle.have_vel or obstacle.have_goal:
                obstacle.set_vel()  # 毎時刻速度を変える
                obstacle.state.p_pos += obstacle.state.p_vel * self.dt    
            
    def __update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      
    
    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.__apply_action_force(p_force)
        # apply environment forces
        p_force = self.__apply_environment_force(p_force)
        # integrate physical state -> ここでポジション更新
        self.__integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.__update_agent_state(agent)
