import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
from multiagent.core import World, Agent, Follower, Obstacle
from multiagent.scenario import BaseScenario
from multiagent.scenarios.base_funcs import Basefuncs
from numpy import linalg as LA
from collections import deque
import copy
import random

class Scenario(BaseScenario):   
    def __init__(self):
        # リーダー数(前方，後方)，フォロワー数，障害物数の設定
        self.num_front_Ls = 0
        self.num_back_Ls = 2
        self.num_Ls = self.num_back_Ls + self.num_front_Ls
        self.num_Fs = 6
        self.num_Os = 1
        self.funcs = Basefuncs()
    
    def make_world(self):
        world = World()
        # add Leaders
        world.agents = [Agent() for i in range(self.num_Ls)]
        for i, L in enumerate(world.agents):
            L.name = 'leader_%d' % i
            L.collide = True
            L.silent = True
            L.front = True if i < self.num_front_Ls else False  # 最初の数体を前方のリーダーとする
            if L.front: L.color = np.array([1, 0.5, 0])
            else: L.color = np.array([1, 0, 0])
        # add Followers
        world.followers = [Follower() for i in range(self.num_Fs)]
        for i, F in enumerate(world.followers):
            F.name = 'follower_%d' % i
            F.collide = True
            F.movable = True
            F.color = np.array([0, 0, 1])
        # add obstacles
        world.obstacles = [Obstacle() for i in range(self.num_Os)]
        for i, O in enumerate(world.obstacles):
            O.name = 'obstacle_%d' % i
            # O.size = 0.1
            O.collide = True
            O.movable = True
            O.color = np.array([0, 0.5, 0])
        self.max_dis_to_des = 15.
        self.max_dis_to_F = 5.
        self.max_dis_to_O = 5.
        # make initial conditions
        self.reset_world(world)
        return world
    
    def reset_world(self, world, is_replay_epi = False):
        # goal到達時の閾値 
        self.rho_g = 1.0
        if self.funcs._make_rand_sign() == 1 or - 1:
            # 目的地の座標
            sign_x = self.funcs._make_rand_sign(); sign_y = self.funcs._make_rand_sign()
            sign_y = 1
            scale = 6
            # self.des = np.array([sign_x * scale + (scale / 3) * (2 * np.random.rand() - 1),
            #                       sign_y * scale + (scale / 3) * np.random.rand()])
            
            self.des = np.array([0, 7.5])
            # replayでない場合のみランダムに初期値を設定する
            if not is_replay_epi:
                self.F_pos = self.funcs._set_F_pos(world) 
                self.front_L_pos = self.funcs._set_front_L_pos(world, self.F_pos, self.num_front_Ls)
                self.back_L_pos = self.funcs._set_back_L_pos_st2(world, self.F_pos, self.num_back_Ls)
                self.O_pos = self.funcs._set_O_pos_st2(world, self.F_pos, self.des)

                # rotate F_pos
                rotate_angle = np.radians(random.randint(-90, 90))
                F_width = world.followers[0].r_F["r4"]
                self.F_pos = self.funcs._rotate_axis(self.F_pos, F_width, rotate_angle)
        else: 
            rand = 2 * np.random.rand()
            angle_list = [-60, -45, -30, 0, 30, 45, 60, 90]
            angle = random.choice(angle_list) * (3.14 / 180)
    

            F_width = world.followers[0].r_F["r4"]
            self.des = np.array([4.884644702315025455e+00 - rand, 4.619343504949231516e+00 + rand])
            self.F_pos = [np.array([3.206951570702668342e+00, 0.0e+00]),
                                 np.array([3.206951570702668342e+00 + F_width * np.sin(-angle), F_width * np.cos(-angle)])]
            for i in range(2):
                next_fol_pos1 = np.array([self.F_pos[2 * i][0] + F_width * np.cos(angle),
                                          self.F_pos[2 * i][1] + F_width * np.sin(angle)])
                next_fol_pos2 = np.array([self.F_pos[2 * i + 1][0] + F_width * np.cos(angle),
                                          self.F_pos[2 * i + 1][1] + F_width * np.sin(angle)])
                self.F_pos.append(next_fol_pos1)
                self.F_pos.append(next_fol_pos2)
            # self.F_pos = [np.array([3.206951570702668342e+00, 0.000000000000000000e+00]),
            #                     np.array([3.906951570702668519e+00, 0.000000000000000000e+00]),
            #                     np.array([4.606951570702667809e+00, 0.000000000000000000e+00]),
            #                     np.array([3.206951570702668342e+00, 6.999999999999999556e-01]),
            #                     np.array([3.906951570702668519e+00, 6.999999999999999556e-01]),
            #                     np.array([4.606951570702667809e+00, 6.999999999999999556e-01])]
            self.front_L_pos = self.funcs._set_front_L_pos(world, self.F_pos, self.num_front_Ls)
            self.back_L_pos = [np.array([3.411556323116248901e+00, -2.500000000000000089e+00]) + 0.5 * (2 * np.random.rand() - 1),
                                    np.array([4.898787262172401569e+00, -2.500000000000000089e+00]) + 0.5 * (2 * np.random.rand() - 1)]
            self.O_pos = [np.array([4.045798136508846454e+00, 2.309671752474615758e+00 + rand])]
        
        if self.funcs._make_rand_sign() == 1:
            tmp = self.back_L_pos[0]
            self.back_L_pos[0] = self.back_L_pos[1]
            self.back_L_pos[1] = tmp
        
        pos_dict = {"follower": copy.deepcopy(self.F_pos), "front_leader": copy.deepcopy(self.front_L_pos),
                    "back_leader": copy.deepcopy(self.back_L_pos), "landmark": copy.deepcopy(self.O_pos)}
        ################################################
        # set leader's random initial states
        for i, L in enumerate(world.agents):
            if L.front: L.state.p_pos = self.front_L_pos[i]
            else: L.state.p_pos = self.back_L_pos[i - self.num_front_Ls]
            L.state.p_vel = np.zeros(world.dim_p)
            L.state.c = np.zeros(world.dim_c)
        # set follower's random initial states
        for i, F in enumerate(world.followers):
            F.state.p_pos = self.F_pos[i]
            F.state.p_vel = np.zeros(world.dim_p)
        # set obstacles' random initial states   
        for i, O in enumerate(world.obstacles):
            O.state.p_pos = self.O_pos[i]
            O.state.p_vel = np.zeros(world.dim_p)

        # 観測用のリストの用意
        self.max_len = 2
        self.L_to_COM_deques = []
        self.L_to_Fs_deques = []
        self.L_to_Ls_deques = []
        self.L_to_Os_deques = []
        for i in range(self.num_Ls):
            L_to_COM_deque = deque(maxlen = self.max_len)
            L_to_Fs_deque = deque(maxlen = self.max_len)
            L_to_Ls_deque = deque(maxlen = self.max_len)
            L_to_Os_deque = deque(maxlen = self.max_len)
            for j in range(self.max_len):
                COM_pos_tmp = []; Fs_pos_tmp = []; Ls_pos_tmp = []; Os_pos_tmp = [];
                COM_pos_tmp.append(np.array([0, 0]))
                for l in range(len(world.agents)-1): 
                    Ls_pos_tmp.append(np.array([0, 0]))
                for f in range(len(world.followers)): 
                    Fs_pos_tmp.append(np.array([0, 0]))
                for o in range(len(world.obstacles)): 
                    Os_pos_tmp.append(np.array([0, 0]))
                L_to_COM_deque.append(COM_pos_tmp)
                L_to_Ls_deque.append(Ls_pos_tmp)
                L_to_Fs_deque.append(Fs_pos_tmp)
                L_to_Os_deque.append(Os_pos_tmp)
            
            self.L_to_COM_deques.append(L_to_COM_deque)
            self.L_to_Fs_deques.append(L_to_Fs_deque)
            self.L_to_Ls_deques.append(L_to_Ls_deque)
            self.L_to_Os_deques.append(L_to_Os_deque)
        
        # 一番遠いフォロワの距離
        self.max_dis_old = 0; self.max_dis_idx = 0; self.max_dis_idx_old = 0

        self.is_goal_old = False
        self.dis_to_des_old = self.funcs._calc_dis_to_des(world, self.des)
        # 分裂の報酬のための変数
        self.is_div_old = False
        # カウンターの用意
        self.counter = 0
        self.dis_deque = deque([0, 0], maxlen = 2)
        # 報酬の用意
        self.R_F_far = 0; self.R_g = 0; self.R_div = 0; self.R_L_close = 0;
        self.R_col = 0; self.R_O = 0; self.R_back = 0
            
        return self.des, self.rho_g, self.funcs._calc_F_COM, pos_dict
    
    def reward(self, L, world):
        if int(L.name.replace('leader_', '')) < self.num_front_Ls + 1:  # 1台目のリーダー
            # 一番遠いフォロワをゴールに近づける報酬
            if L.front: self.R_F_far = 0
            else:
                Fs_dis_to_des = np.array(self.funcs._calc_Fs_dis_to_des(world, self.des))
                max_dis = Fs_dis_to_des.max(); self.max_dis_idx =  np.argmax(Fs_dis_to_des)
                if self.max_dis_idx == self.max_dis_idx_old:
                    # 1stepあたりのmaxの移動量で正規化
                    if (self.max_dis_old - max_dis) > 1e-4: self.R_F_far = 5 * (self.max_dis_old - max_dis)
                    else: self.R_F_far = 0
                else: self.R_F_far = 0
                self.max_dis_old = max_dis; self.max_dis_idx_old = self.max_dis_idx
                
            # goalまでの距離に関する報酬
            dis_to_des = self.funcs._calc_dis_to_des(world, self.des)
            if L.front:  # 前方のエージェント -> 衝突回避の報酬を大きく
                if (self.dis_to_des_old - dis_to_des) > 1e-4 : self.R_g = 20 * (self.dis_to_des_old - dis_to_des)
                else: self.R_g = - 1
                self.R_g = 0
            else: # 後方のエージェント -> ゴールに近づく報酬を大きく
                if dis_to_des > 2.0:
                    self.R_g = 5 * (self.dis_to_des_old - dis_to_des)
                else:
                    if (self.dis_to_des_old - dis_to_des) > 1e-4:
                        self.R_g = 1 / (dis_to_des + 1)  # denseな報酬
                    else: self.R_g = -0.1
            is_goal = self.funcs._check_goal(dis_to_des, self.rho_g)
            if is_goal: self.R_g = 30
            # 値の更新
            self.dis_to_des_old = dis_to_des
            
            # 分裂に関する報酬
            is_div = self.funcs._chech_div(world)
            if not is_div and not self.is_div_old: self.R_div = 0  # 分裂していない状態が続くとき
            elif is_div and self.is_div_old: self.R_div = 0  # 分裂している状態が続くとき
            elif is_div and not self.is_div_old: self.R_div = - 20  # 分裂したとき
            else: self.R_div = 5  # 分裂から復帰したとき
            # 値の更新
            self.is_div_old = is_div
            
            # リーダーがフォロワから離れすぎないための報酬
            min_dis_to_F = self.funcs._calc_min_dis_to_F(L, world)
            if min_dis_to_F > world.followers[0].r_L["r5d"] * 2.0:
               self.R_L_close = - 0.25 * (min_dis_to_F - world.followers[0].r_L["r5d"])
            else: self.R_L_close = 0
               
            # リーダーが後ろ側に回り込むための報酬
            L_dis_to_des = LA.norm(self.des - L.state.p_pos)
            if L_dis_to_des < dis_to_des:
                self.R_back = -0.5 * (dis_to_des - L_dis_to_des)
            else: self.R_back = 0
            ######################## for ensure R_back #######################
            # self.R_back = 0
            
            # 障害物に近づきすぎないための報酬
            if world.obstacles:
                min_dis_to_O = self.funcs._calc_Fs_min_dis_to_O(world)
                if min_dis_to_O < world.followers[0].r_F["r5"] * 1.25:
                    self.R_O = - 0.5 / (min_dis_to_O + 1)
                else: self.R_O = 0
            else: self.R_O = 0
            
            # 衝突に関する報酬
            is_col = self.funcs._check_col(world)
            if L.front:  # 前方のエージェント -> 衝突回避の報酬を大きく
                if is_col: self.R_col = -10
                else: self.R_col = 0
            else:
                if is_col: self.R_col = - 10
                else: self.R_col = 0
        else:  # 後ろのリーダーの二台目以降
            # リーダーがフォロワから離れすぎないための報酬のみ更新
            min_dis_to_F = self.funcs._calc_min_dis_to_F(L, world)
            if min_dis_to_F > world.followers[0].r_L["r5d"] * 2.0:
                self.R_L_close = - 0.25 * (min_dis_to_F - world.followers[0].r_L["r5d"])
            else: self.R_L_close = 0
            
            dis_to_des = self.funcs._calc_dis_to_des(world, self.des)
            # リーダーが後ろ側に回り込むための報酬 
            L_dis_to_des = LA.norm(self.des - L.state.p_pos)  
            if L_dis_to_des < dis_to_des:
                self.R_back = - 0.5 * (dis_to_des - L_dis_to_des)
            else: self.R_back = 0 
            ######################## for ensure R_back #######################
            # self.R_back = 0
        
        reward = self.R_F_far + self.R_g + self.R_div + self.R_L_close + self.R_back + self.R_O + self.R_col
        # to do: 一台ずつ報酬表示できるようにする
        reward_list = np.round(np.array([self.R_F_far, self.R_g, self.R_div, self.R_L_close, self.R_back, self.R_O, self.R_col])\
                    / len(world.agents), decimals=2)
        
        return reward, reward_list
        
    def observation(self, L, world):
        i = int(L.name.replace('leader_', ''))
        
        F_COM = self.funcs._calc_F_COM(world)
        COM_to_des = self.des - F_COM
        COM_to_des = self.funcs._coordinate_trans(COM_to_des)
        COM_to_des[0] /= self.max_dis_to_des  # 距離の正規化

        COM_to_O = world.obstacles[0].state.p_pos - F_COM
        COM_to_O = self.funcs._coordinate_trans(COM_to_O)
        
        # L_to_COM = F_COM - L.state.p_pos
        # L_to_COM = self.funcs._coordinate_trans(L_to_COM)
        # self.L_to_COM_deques[i].append([L_to_COM])
        
        L_to_Ls = []
        for other in world.agents:
            if L is other: continue 
            L_to_L = other.state.p_pos - L.state.p_pos
            L_to_L = self.funcs._coordinate_trans(L_to_L)
            L_to_Ls.append(L_to_L)
        self.L_to_Ls_deques[i].append(L_to_Ls)
        
        L_to_Fs = []
        for F in world.followers:
            L_to_F = F.state.p_pos - L.state.p_pos
            L_to_F = self.funcs._coordinate_trans(L_to_F)
            L_to_Fs.append(L_to_F)    
        self.L_to_Fs_deques[i].append(L_to_Fs)
        
        L_to_Os = []
        for O in world.obstacles: 
            L_to_O = O.state.p_pos - L.state.p_pos
            L_to_O = self.funcs._coordinate_trans(L_to_O)
            L_to_Os.append(L_to_O)
        self.L_to_Os_deques[i].append(L_to_Os)
        
        
        # obs = np.concatenate([COM_to_des] + [agent.state.p_vel] \
        #     + self.L_to_COM_deques[i][0] + self.L_to_COM_deques[i][1] + self.L_to_COM_deques[i][2]\
        #     + self.L_to_Fs_deques[i][0] + self.L_to_Fs_deques[i][1] + self.L_to_Fs_deques[i][2]\
        #     + self.L_to_Ls_deques[i][0] + self.L_to_Ls_deques[i][1] + self.L_to_Ls_deques[i][2]\
        #     + self.L_to_Os_deques[i][0] + self.L_to_Os_deques[i][1] + self.L_to_Os_deques[i][2])
        
        obs = np.concatenate([COM_to_des] + [COM_to_O] + [L.state.p_vel] \
            + self.L_to_Fs_deques[i][0] + self.L_to_Fs_deques[i][1]\
            + self.L_to_Ls_deques[i][0] + self.L_to_Ls_deques[i][1])
        
        return obs
    
    def check_done(self, L, world):
        # goalしたか否か
        dis_to_des = self.funcs._calc_dis_to_des(world, self.des)
        is_goal = self.funcs._check_goal(dis_to_des, self.rho_g)
        # # 衝突が起こっているか否か
        # is_col = self.funcs._check_col(world)
        # 分裂したか否か
        is_div = self.funcs._chech_div(world)
        
        # desまでの距離がmaxを超えていないか
        dis_to_des = self.funcs._calc_dis_to_des(world, self.des)
        if dis_to_des > self.max_dis_to_des: is_exceed = True
        else: is_exceed = False

        if is_goal or is_div or is_exceed: return True
        else: return False