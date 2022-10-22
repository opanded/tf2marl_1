import numpy as np
from tf2marl.multiagent.core import World, Agent, Follower, Landmark
from tf2marl.multiagent.scenario import BaseScenario
from numpy import linalg as LA
from sklearn.cluster import DBSCAN
from collections import deque
import copy

class Scenario(BaseScenario):   
    def __init__(self):
        # 学習時のエピソードのステップ数
        # self.max_episode_len = arglist.max_episode_len
        # リーダー数(前方，後方)，フォロワー数，障害物数の設定
        self.num_front_leaders = 0
        self.num_back_leaders = 2
        self.num_leaders = self.num_back_leaders + self.num_front_leaders
        self.num_followers = 6
        self.num_landmarks = 1
    
    def make_world(self):
        # world = World(self.max_episode_len)
        world = World()
        # add Leaders
        world.agents = [Agent() for i in range(self.num_leaders)]
        world.n_adversaries = self.num_front_leaders
        for i, agent in enumerate(world.agents):
            agent.name = 'leader_%d' % i
            agent.collide = False
            agent.silent = True
            agent.front = True if i < self.num_front_leaders else False # 最初の数体を前方のリーダーとする。
        # add Followers
        world.followers = [Follower() for i in range(self.num_followers)]
        for i, follower in enumerate(world.followers):
            follower.name = 'follower_%d' % i
            follower.collide = False
            follower.movable = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark_%d' % i
            landmark.size = 0.1
            landmark.collide = False
            landmark.movable = True
        # make initial conditions
        self.reset_world(world)
        return world
    
    def reset_world(self, world, is_replay_epi = False):
        # 目的地の座標 -> エピソード毎にランダムにする
        rand = np.random.rand()
        if rand <= 0.5: sign_x = 1
        else: sign_x = -1
        rand = np.random.rand()
        if rand <= 0.5: sign_y = 1
        else: sign_y = -1
        # sign_y = 1
        self.dest = np.array([sign_x * 3 + 1 * (2 * np.random.rand() - 1), sign_y * 3 + 1 * (2 * np.random.rand() - 1)])
        # self.dest = np.array([3.450131225716176253e+00, 2.422564031445024302e+00])
        # goal到達時の閾値 
        self.rho_g = 0.5
        
        # replayでない場合のみランダムに初期値を設定する
        if not is_replay_epi:
            self.follower_pos = self.__set_follower_pos(world) 
            self.front_leader_pos = self.__set_front_leader_pos(world, self.follower_pos)
            self.back_leader_pos = self.__set_back_leader_pos(world, self.follower_pos)
            self.landmark_pos = self.__set_landmark_pos(world, self.follower_pos)
        
        # 一定の確率でリーダーのポジションを入れ替える
        if self.__make_random_sign() == 1:
                tmp = self.back_leader_pos[0]
                self.back_leader_pos[0] = self.back_leader_pos[1]
                self.back_leader_pos[1] = tmp
        
        pos_dict = {"follower": copy.deepcopy(self.follower_pos), "front_leader": copy.deepcopy(self.front_leader_pos),
                    "back_leader": copy.deepcopy(self.back_leader_pos), "landmark": copy.deepcopy(self.landmark_pos),
                    "dest": [copy.deepcopy(self.dest)]}
        
        # set leader's random initial states
        for i, agent in enumerate(world.agents):
            if agent.front: agent.state.p_pos = self.front_leader_pos[i]
            else: agent.state.p_pos = self.back_leader_pos[i - self.num_front_leaders]
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # set follower's random initial states
        for i, follower in enumerate(world.followers):
            follower.state.p_pos = self.follower_pos[i]
            follower.state.p_vel = np.zeros(world.dim_p)
        # set landmarks' random initial states   
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = self.landmark_pos[i]
            landmark.state.p_vel = np.zeros(world.dim_p)
            
        # random properties for leaders, followers, landmarks
        for i, agent in enumerate(world.agents):
            if agent.front: agent.color = np.array([1, 0.5, 0])
            else: agent.color = np.array([1, 0, 0])
        for i, follower in enumerate(world.followers):
            follower.color = np.array([0, 0, 1])
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0, 1, 0])
            
        # 観測用のリストの用意
        self.max_len = 3
        self.L_to_COM_deques = []
        self.L_to_Fs_deques = []
        self.L_to_Ls_deques = []
        self.L_to_obstacles_deques = []
        for i in range(self.num_leaders):
            L_to_COM_deque = deque(maxlen = self.max_len)
            L_to_Fs_deque = deque(maxlen = self.max_len)
            L_to_Ls_deque = deque(maxlen = self.max_len)
            L_to_obstacles_deque = deque(maxlen = self.max_len)
            for j in range(self.max_len):
                COM_pos_tmp = []; Fs_pos_tmp = []; Ls_pos_tmp = []; Landmarks_pos_tmp = [];
                COM_pos_tmp.append(np.array([0, 0]))
                for l in range(len(world.agents)-1): 
                    Ls_pos_tmp.append(np.array([0, 0]))
                for f in range(len(world.followers)): 
                    Fs_pos_tmp.append(np.array([0, 0]))
                for o in range(len(world.landmarks)): 
                    Landmarks_pos_tmp.append(np.array([0, 0]))
                L_to_COM_deque.append(COM_pos_tmp)
                L_to_Ls_deque.append(Ls_pos_tmp)
                L_to_Fs_deque.append(Fs_pos_tmp)
                L_to_obstacles_deque.append(Landmarks_pos_tmp)
            
            self.L_to_COM_deques.append(L_to_COM_deque)
            self.L_to_Fs_deques.append(L_to_Fs_deque)
            self.L_to_Ls_deques.append(L_to_Ls_deque)
            self.L_to_obstacles_deques.append(L_to_obstacles_deque)
        
        # 一番遠いフォロワの距離
        self.max_dist_old = 0; self.max_dist_idx = 0; self.max_dist_idx_old = 0

        self.is_goal_old = False
        self.dist_to_dest_old = self.__calc_dist_to_dest(world)
        # 分裂の報酬のための変数
        self.is_deivision_old = False
        # カウンターの用意
        self.counter = 0
        self.dist_deque = deque([0, 0], maxlen = 2)
        # 報酬の用意
        self.R_F_far = 0; self.R_g = 0; self.R_div = 0; self.R_L_close = 0;
        self.R_col = 0; self.R_obs = 0; self.R_back = 0
            
        return self.dest, self.rho_g, self.__calc_F_COM, pos_dict
    
    def reward(self, agent, world):
        if int(agent.name.replace('leader_', '')) < self.num_front_leaders + 1:  # 1台目のリーダー
            # 一番遠いフォロワをゴールに近づける報酬
            if agent.front: self.R_F_far = 0
            else:
                Fs_dist_to_dest = np.array(self.__calc_Fs_dist_to_dest(world))
                max_dist = Fs_dist_to_dest.max(); self.max_dist_idx =  np.argmax(Fs_dist_to_dest)
                if self.max_dist_idx == self.max_dist_idx_old:
                    # 1stepあたりのmaxの移動量で正規化
                    if (self.max_dist_old - max_dist) > 1e-4: self.R_F_far = 5 * (self.max_dist_old - max_dist)
                    else: self.R_F_far = 0
                else: self.R_F_far = 0
                self.max_dist_old = max_dist; self.max_dist_idx_old = self.max_dist_idx
            
            # goalまでの距離に関する報酬
            dist_to_dest = self.__calc_dist_to_dest(world)
            if agent.front:  # 前方のエージェント -> 衝突回避の報酬を大きく
                if (self.dist_to_dest_old - dist_to_dest) > 1e-4 : self.R_g = 20 * (self.dist_to_dest_old - dist_to_dest)
                else: self.R_g = - 1
                self.R_g = 0
            else: # 後方のエージェント -> ゴールに近づく報酬を大きく
                if dist_to_dest > 2.0:
                    self.R_g = 5 * (self.dist_to_dest_old - dist_to_dest)
                else:
                    if (self.dist_to_dest_old - dist_to_dest) > 1e-4:
                        self.R_g = 1 / (dist_to_dest + 1)  # denseな報酬
                    else: self.R_g = -0.25
            is_goal = self.__check_goal(dist_to_dest)
            if is_goal: self.R_g = 50
            # 値の更新
            self.dist_to_dest_old = dist_to_dest
            
            # 分裂に関する報酬
            is_division = self.__chech_division(world)
            if not is_division and not self.is_deivision_old: self.R_div = 0  # 分裂していない状態が続くとき
            elif is_division and self.is_deivision_old: self.R_div = 0  # 分裂している状態が続くとき
            elif is_division and not self.is_deivision_old: self.R_div = - 30  # 分裂したとき
            else: self.R_div = 5  # 分裂から復帰したとき
            # 値の更新
            self.is_deivision_old = is_division
            
            # リーダーがフォロワから離れすぎないための報酬
            min_dist_to_F = self.__calc_min_dist_to_F(agent, world)
            if min_dist_to_F > world.followers[0].r_L["r5d"] * 2.0:
               self.R_L_close = - 0.25 * (min_dist_to_F - world.followers[0].r_L["r5d"])
            else: self.R_L_close = 0
            self.R_L_close = 0
               
            # リーダーが後ろ側に回り込むための報酬
            # L_dist_to_dest = LA.norm(self.dest - agent.state.p_pos)
            # if L_dist_to_dest < dist_to_dest * 1.2:
            #     self.R_back = -0.5 * (dist_to_dest * 1.2 - L_dist_to_dest)
            # else: self.R_back = 0
            
            G_to_far_fol = world.followers[self.max_dist_idx].state.p_pos - self.dest
            far_fol_to_L = agent.state.p_pos -world.followers[self.max_dist_idx].state.p_pos
            if (LA.norm(far_fol_to_L) < world.followers[0].r_L["r5d"] * 2.0) and np.dot(G_to_far_fol, far_fol_to_L) >= 0:
                self.R_back = 0.1 * (world.followers[0].r_L["r5d"] * 2.0 - LA.norm(far_fol_to_L))
            else: self.R_back = -0.2
                
            # 障害物に近づきすぎないための報酬
            if world.landmarks:
                min_dist_to_obs = self.__calc_Fs_min_dist_to_obs(world)
                if min_dist_to_obs < world.followers[0].r_F["r5"] * 1.5:
                    self.R_obs = - 1 / (min_dist_to_obs + 1)
                else: self.R_obs = 0
            else: self.R_obs = 0
            
            # 衝突に関する報酬
            is_collision = self.__check_collision(world)
            if agent.front:  # 前方のエージェント -> 衝突回避の報酬を大きく
                if is_collision: self.R_col = -10
                else: self.R_col = 0
            else:
                if is_collision: self.R_col = - 20
                else: self.R_col = 0
        else:  # 後ろのリーダーの二台目以降
            # リーダーがフォロワから離れすぎないための報酬のみ更新
            min_dist_to_F = self.__calc_min_dist_to_F(agent, world)
            if min_dist_to_F > world.followers[0].r_L["r5d"] * 2.0:
                self.R_L_close = - 0.25 * (min_dist_to_F - world.followers[0].r_L["r5d"])
            else: self.R_L_close = 0
            self.R_L_close = 0
            
            # dist_to_dest = self.__calc_dist_to_dest(world)
            # リーダーが後ろ側に回り込むための報酬 
            # L_dist_to_dest = LA.norm(self.dest - agent.state.p_pos)  
            # if L_dist_to_dest < dist_to_dest * 1.2:
            #     self.R_back = -0.5 * (dist_to_dest * 1.2 - L_dist_to_dest)
            # else: self.R_back = 0 
            
            G_to_far_fol = world.followers[self.max_dist_idx].state.p_pos - self.dest
            far_fol_to_L = agent.state.p_pos -world.followers[self.max_dist_idx].state.p_pos
            if (LA.norm(far_fol_to_L) < world.followers[0].r_L["r5d"] * 2.0) and np.dot(G_to_far_fol, far_fol_to_L) >= 0:
                self.R_back = 0.1 * (world.followers[0].r_L["r5d"] * 2.0 - LA.norm(far_fol_to_L))
            else:
                self.R_back = - 0.2
        
        reward = self.R_F_far + self.R_g + self.R_div + self.R_L_close + self.R_back + self.R_obs + self.R_col
        # to do: 一台ずつ報酬表示できるようにする
        reward_list = np.round(np.array([self.R_F_far, self.R_g, self.R_div, self.R_L_close, self.R_back, self.R_obs, self.R_col])\
                    / len(world.agents), decimals=2)
        
        return reward, reward_list
        
    def observation(self, agent, world):
        i = int(agent.name.replace('leader_', ''))
        
        F_COM = self.__calc_F_COM(world)
        COM_to_dest = self.dest - F_COM
        
        L_to_COM = F_COM - agent.state.p_pos
        self.L_to_COM_deques[i].append([L_to_COM])
        
        L_to_Ls = []
        for other in world.agents:
            if agent is other: continue 
            L_to_L = other.state.p_pos - agent.state.p_pos
            L_to_Ls.append(L_to_L)
        self.L_to_Ls_deques[i].append(L_to_Ls)
        
        L_to_Fs = []
        for follower in world.followers:
            L_to_F = follower.state.p_pos - agent.state.p_pos
            L_to_Fs.append(L_to_F)    
        self.L_to_Fs_deques[i].append(L_to_Fs)
            
        
        L_to_obstacles = []
        for land in world.landmarks: 
            L_to_obstacle = land.state.p_pos - agent.state.p_pos
            L_to_obstacles.append(L_to_obstacle)
        self.L_to_obstacles_deques[i].append(L_to_obstacles)
        
        
        obs = np.concatenate([COM_to_dest] + [agent.state.p_vel] \
            + self.L_to_COM_deques[i][0] + self.L_to_COM_deques[i][1] + self.L_to_COM_deques[i][2]\
            + self.L_to_Fs_deques[i][0] + self.L_to_Fs_deques[i][1] + self.L_to_Fs_deques[i][2]\
            + self.L_to_Ls_deques[i][0] + self.L_to_Ls_deques[i][1] + self.L_to_Ls_deques[i][2]\
            + self.L_to_obstacles_deques[i][0] + self.L_to_obstacles_deques[i][1] + self.L_to_obstacles_deques[i][2])
        
        return obs
    
    def check_done(self, agent, world):
        # goalしたか否か
        dist_to_dest = self.__calc_dist_to_dest(world)
        is_goal = self.__check_goal(dist_to_dest)
        # 衝突が起こっているか否か
        is_collision = self.__check_collision(world)
        # 分裂したか否か
        is_division = self.__chech_division(world)

        # if is_goal or is_collision or is_division: return True
        # 衝突しても学習を続ける
        if is_goal or is_division: return True
        else: return False
    
###########################################  For reset_world ###########################################
    def __make_random_sign(self):
        if np.random.rand() <= 0.5: return 1
        else: return -1
    
    def __set_follower_pos(self, world):
        follower_pos = []
        # 3台ずつ列状に並べる
        n = 3
        m = -(-len(world.followers) // n) # m: 列数，演算子を用いて切り上げをしている
        follower_ref_coordinate = np.array([5 * np.random.rand(), 0])
        follower_width = world.followers[0].r_F["r4"]
        for i in range(m): 
            for j in range(n):
                follower_next_coordinate = np.array([follower_ref_coordinate[0] + j * follower_width, i * follower_width]) 
                follower_pos.append(follower_next_coordinate)
        
        # follower_pos = [np.array([1.942286854877933733e-01, 0.000000000000000000e+00]),
        #                 np.array([8.942286854877933289e-01, 0.000000000000000000e+00]),
        #                 np.array([1.594228685487793395e+00, 0.000000000000000000e+00]),
        #                 np.array([1.942286854877933733e-01, 6.999999999999999556e-01]),
        #                 np.array([8.942286854877933289e-01, 6.999999999999999556e-01]),
        #                 np.array([1.594228685487793395e+00, 6.999999999999999556e-01])]
        return follower_pos     

    def __set_back_leader_pos(self, world, follower_pos):
        back_leader_pos = []
        # 一定の確率でリーダーをランダムに配置
        if np.random.rand() <= 0.5:
          # 右側後方のフォロワを基準にする
          back_leader_ref_coordinate = np.array([follower_pos[0][0], follower_pos[0][1]])
          leader_width = world.followers[0].r_L["r5d"]
          # 台数分の初期位置を定義する
          for i in range(self.num_back_leaders):
              back_leader_next_coordinate = np.array([back_leader_ref_coordinate[0] + i * leader_width + np.random.rand()\
                                                      ,back_leader_ref_coordinate[1] - leader_width * 1.1]) 
              back_leader_pos.append(back_leader_next_coordinate)
        else: 
          # 左側前方のフォロワを基準にする
          back_leader_ref_coordinate = np.array([follower_pos[-1][0], follower_pos[-1][1]])
          leader_width = world.followers[0].r_L["r5d"]
          # 台数分の初期位置を定義する
          for i in range(self.num_back_leaders):
              back_leader_next_coordinate = np.array([back_leader_ref_coordinate[0] - i * leader_width + np.random.rand()\
                                                      ,back_leader_ref_coordinate[1] + leader_width * 1.1]) 
              back_leader_pos.append(back_leader_next_coordinate)
        
        # back_leader_pos = [np.array([2.543700918185702431e+00, 1.800000000000000044e+00]),
        #                 np.array([1.488687429866359935e+00, 1.800000000000000044e+00])]
        return back_leader_pos
    
    def __set_front_leader_pos(self, world, follower_pos):
        front_leader_pos = []
        # 左側前方のフォロワを基準にする
        front_leader_ref_coordinate = np.array([follower_pos[-1][0], follower_pos[-1][1]])
        leader_width = world.followers[0].r_L["r5d"]
        # 台数分の初期位置を定義する
        for i in range(self.num_front_leaders):
            front_leader_next_coordinate = np.array([front_leader_ref_coordinate[0] - i * leader_width + np.random.rand()\
                                                    ,front_leader_ref_coordinate[1] + leader_width * 1.1]) 
            front_leader_pos.append(front_leader_next_coordinate)
        
        return front_leader_pos
    
    def __set_landmark_pos(self, world, follower_pos):
        # フォロワと目的地の中点
        # landmark_ref_coordinate = follower_pos[0] + 2 * (self.dest - follower_pos[0]) / 3
        # landmark_ref_coordinate = follower_pos[0] + (self.dest - follower_pos[0]) / 2
        landmark_ref_coordinate = follower_pos[0] + 10
        landmark_pos = []
        for i in range(len(world.landmarks)):
            landmark_next_coordinate = np.array([landmark_ref_coordinate[0] + i * 1, landmark_ref_coordinate[1]]) 
            landmark_pos.append(landmark_next_coordinate)
        
        # landmark_pos = [np.array([1.019422868548779348e+01, 1.000000000000000000e+01])]
        return landmark_pos
    
    # return all agents that are back of swarm
    def back_agents(self, world):
        return [agent for agent in world.agents if not agent.front]

    # return all front of swarm
    def front_agents(self, world):
        return [agent for agent in world.agents if agent.front]

###########################################  For reward, observation, check_done ###########################################
    
    def __calc_min_dist_to_F(self, agent, world):
        min_dist = 10000
        # followerとの距離取得してminの距離を返す
        for follower in world.followers:
            delta_pos = follower.state.p_pos - agent.state.p_pos
            dist = LA.norm(delta_pos)
            if dist < min_dist: min_dist = dist
        
        return min_dist
    
    def __calc_Fs_dist_to_COM(self, world):
        follower_COM = self.__calc_F_COM(world)
        Fs_dist_to_COM = []
        for follower in world.followers:
            dist_to_COM = LA.norm(follower_COM - follower.state.p_pos)
            Fs_dist_to_COM.append(dist_to_COM)

        return Fs_dist_to_COM
    
    def __calc_Fs_dist_to_dest(self, world):
        Fs_dist_to_dest = []
        for follower in world.followers:
            F_dist_to_dest = LA.norm(self.dest - follower.state.p_pos)
            Fs_dist_to_dest.append(F_dist_to_dest)

        return Fs_dist_to_dest
    
    def __calc_dist_to_dest(self, world):
        follower_COM = self.__calc_F_COM(world)
        dist_to_dest = LA.norm(follower_COM - self.dest)
        
        return dist_to_dest
    
    def __calc_Fs_min_dist_to_obs(self, world):
        min_dist_to_obs = 1e5
        for follower in world.followers:
            F_dist_to_obs = LA.norm(world.landmarks[0].state.p_pos - follower.state.p_pos)
            if F_dist_to_obs < min_dist_to_obs: min_dist_to_obs = F_dist_to_obs

        return min_dist_to_obs
    
    def __calc_F_COM(self, world):
        follower_sum = np.array([0.,0.])
        for follower in world.followers:
            follower_sum += follower.state.p_pos
        follower_COM = follower_sum / len(world.followers)
        
        return follower_COM
    
    def __check_collision(self, world):
        is_collision = False
        # obj_iはリーダー，フォロワ，障害物のすべてを含む
        for obj_i in world.entities:
            for obj_j in world.entities:
                if obj_i is obj_j: continue
                delta_pos = obj_j.state.p_pos - obj_i.state.p_pos
                dist = LA.norm(delta_pos)
                dist_min = obj_i.size + obj_i.size
                if dist < dist_min:
                    is_collision = True
                    break
            if is_collision: break
        
        return is_collision
    
    def __check_goal(self, dist_to_dest):
        if (dist_to_dest <= self.rho_g): is_goal = True
        else: is_goal = False
        
        return is_goal
    
    def __chech_division(self, world):
        Fs_pos = []
        for follower in world.followers:
            Fs_pos.append(follower.state.p_pos)
        
        dbscan = DBSCAN(eps = world.followers[0].r_F["r5"], min_samples = 2).fit(Fs_pos)
        y_dbscan = dbscan.labels_
        
        if len(np.where(y_dbscan ==-1)[0]) > 0 or np.max(y_dbscan) != 0: return True
        else: return False