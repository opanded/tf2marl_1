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
        self.num_Os = random.choice([1, 2])
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
            else: 
                if i == 0: L.color = np.array([1, 0, 0])
                else: L.color = np.array([1, 0.5, 0])
        # add Followers
        world.followers = [Follower() for i in range(self.num_Fs)]
        for i, F in enumerate(world.followers):
            F.name = 'follower_%d' % i
            F.collide = True
            F.movable = False
            F.color = np.array([0, 0, 1])
        self.max_dis_to_des = 15.
        self.max_dis_to_L = 7.5
        self.max_dis_to_F = 7.5
        self.max_dis_to_O = 5. 
        # リーダー入れ替え用の配列を用意
        self.rand_idx = [i for i in range(self.num_Ls)]
        
        # make initial conditions
        self.reset_world(world)
        return world
    
    def reset_world(self, world):
        # 障害物の数をエピソード毎に変化させる
        self.num_Os = random.choice([1, 2])
        # add obstacles
        world.obstacles = [Obstacle() for i in range(self.num_Os)]
        
        # goal到達時の閾値 
        self.rho_g = 1.0
        if self.funcs._make_rand_sign() == 1 or -1:
            # 目的地の座標
            self.des = np.array([0, 7.5])
            self.F_pos = self.funcs._set_F_pos(world) 
            self.front_L_pos = self.funcs._set_front_L_pos(world, self.F_pos, self.num_front_Ls)
            self.back_L_pos = self.funcs._set_back_L_pos_st2(world, self.F_pos, self.num_back_Ls)
            self.O_pos = self.funcs._set_O_pos_st2(world, self.F_pos, self.des)

            # rotate F_pos
            # rotate_angle = np.radians(random.randint(-90, 90))
            # F_width = world.followers[0].r_F["r4"]
            # self.F_pos = self.funcs._rotate_axis(self.F_pos, F_width, rotate_angle)
        else: 
            rand = 2 * np.random.rand()
            angle_list = [-60, -45, -30, 0, 30, 45, 60, 90]
            # angle = random.choice(angle_list) * (3.14 / 180)
            angle = 0

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
        
        # 一定の確率でリーダーのポジションを入れ替える
        L_i = random.choice(self.rand_idx); L_j = random.choice(self.rand_idx)
        if L_i != L_j:
           tmp = self.back_L_pos[L_i]
           self.back_L_pos[L_i] = self.back_L_pos[L_j]
           self.back_L_pos[L_j] = tmp
        
        pos_dict = {"follower": copy.deepcopy(self.F_pos), "front_leader": copy.deepcopy(self.front_L_pos),
                    "back_leader": copy.deepcopy(self.back_L_pos), "obstacle": copy.deepcopy(self.O_pos),
                    "dest": [copy.deepcopy(self.des)]}
        
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
        idx = random.choice([0, 1])
        for i, O in enumerate(world.obstacles):
            O.state.p_pos = self.O_pos[i]
            if i == idx: O.have_vel = True
            if O.have_vel:
                O.state.p_vel = np.array([0.2, 0.2])
            O.name = 'obstacle_%d' % i
            # O.size = 0.1
            O.collide = True
            O.movable = False
            O.color = np.array([0, 0.5, 0])
    
        # 観測用のリストの用意
        self.max_len = 2
        self.COM_to_Os_deques = []
        self.L_to_Fs_deques = []
        self.L_to_Ls_deques = []
        self.L_to_Os_deques = []
        for i in range(self.num_Ls):
            COM_to_O_deque = deque(maxlen = self.max_len)
            L_to_Fs_deque = deque(maxlen = self.max_len)
            L_to_Ls_deque = deque(maxlen = self.max_len)
            L_to_Os_deque = deque(maxlen = self.max_len)
            for j in range(self.max_len):
                Fs_pos_tmp = []; Ls_pos_tmp = []; Os_pos_tmp = []
                for l in range(len(world.agents)-1): 
                    Ls_pos_tmp.append(np.array([0, 0]))
                for f in range(len(world.followers)): 
                    Fs_pos_tmp.append(np.array([0, 0]))
                for o in range(len(world.obstacles)): 
                    Os_pos_tmp.append(np.array([0, 0]))
                COM_to_O_deque.append(Os_pos_tmp)
                L_to_Ls_deque.append(Ls_pos_tmp)
                L_to_Fs_deque.append(Fs_pos_tmp)
                L_to_Os_deque.append(Os_pos_tmp)
            
            self.COM_to_Os_deques.append(COM_to_O_deque)
            self.L_to_Fs_deques.append(L_to_Fs_deque)
            self.L_to_Ls_deques.append(L_to_Ls_deque)
            self.L_to_Os_deques.append(L_to_Os_deque)
        
        # for reward
        # ゴールから一番遠いフォロワの距離
        self.max_dis_old = 0; self.max_dis_idx_old = 0
        # 重心からゴールまでの距離
        self.dis_to_des_old = self.funcs._calc_dis_to_des(world, self.des)
        # 各リーダーから一番近いフォロワまでの距離
        self.min_dis_to_F_old = [self.funcs._calc_min_dis_to_F(L, world) for L in world.agents]
        # 各障害物用の変数
        self.is_close_to_Os = []; self.min_dis_to_Os_old = []
        for _ in range(self.num_Os):
            self.is_close_to_Os.append(False)
            self.min_dis_to_Os_old.append(self.funcs._calc_Fs_min_dis_to_O(world))
        # 衝突判定用の変数
        self.is_col_old = [False for _ in range(self.num_Ls)]
        
        # for observation
        # フォロワの重心位置
        self.F_COM_old = self.funcs._calc_F_COM(world)
        # 各障害物の位置
        self.Os_old = [O.state.p_pos for O in world.obstacles]
        # 1stepの移動の最大値
        self.max_moving_dis = 2 * world.agents[0].max_speed * world.dt
            
        return self.des, self.rho_g, self.funcs._calc_F_COM, pos_dict
    
    def reward(self, L, world):
        if int(L.name.replace('leader_', '')) < self.num_front_Ls + 1:  # 1台目のリーダー
            # 一番遠いフォロワをゴールに近づける報酬
            Fs_dis_to_des = np.array(self.funcs._calc_Fs_dis_to_des(world, self.des))
            max_dis = Fs_dis_to_des.max(); self.max_dis_idx =  np.argmax(Fs_dis_to_des)
            # if self.max_dis_idx == self.max_dis_idx_old:
            if (self.max_dis_old - max_dis) > 1e-4: self.R_F_far = 5 * (self.max_dis_old - max_dis)
            else: self.R_F_far = 0
            # else: self.R_F_far = 0
            self.max_dis_old = max_dis 
            # self.max_dis_idx_old = self.max_dis_idx
            
            # goalまでの距離に関する報酬
            self.dis_to_des = self.funcs._calc_dis_to_des(world, self.des)
            if self.dis_to_des > 2.0:
                self.R_g = 5 * (self.dis_to_des_old - self.dis_to_des)
            else:
                if (self.dis_to_des_old - self.dis_to_des) > 1e-4:
                    self.R_g = 5 * (self.dis_to_des_old - self.dis_to_des)  # denseな報酬
                else: self.R_g = -0.1
            self.is_goal = self.funcs._check_goal(self.dis_to_des, self.rho_g)
            if self.is_goal: self.R_g = 30
            # 値の更新
            self.dis_to_des_old = self.dis_to_des
            
            # 分裂に関する報酬
            self.is_div = self.funcs._chech_div(world)
            if not self.is_div: self.R_div = 0
            else: self.R_div = -20
            
            # リーダーがフォロワから離れすぎないための報酬
            self.min_dis_to_F = self.funcs._calc_min_dis_to_F(L, world)
            if self.min_dis_to_F > world.followers[0].r_L["r5d"] * 2.0:
                if self.min_dis_to_F > self.min_dis_to_F_old[0]:
                    self.R_L_close = - 0.05 * (self.min_dis_to_F - world.followers[0].r_L["r5d"])
                else: self.R_L_close = 0
            else: self.R_L_close = 0
            self.min_dis_to_F_old[0] = self.min_dis_to_F
               
            # リーダーが後ろ側に回り込むための報酬
            L_dis_to_des = LA.norm(self.des - L.state.p_pos)
            if L_dis_to_des < self.dis_to_des:
                self.R_back = - 0.05 * (self.dis_to_des - L_dis_to_des)
            else: self.R_back = 0
            
            # 障害物に近づきすぎないための報酬
            if world.obstacles:
                self.R_O = 0
                min_dis_to_Os = self.funcs._calc_Fs_min_dis_to_O(world)
                for idx, min_dis_to_O in enumerate(min_dis_to_Os):
                    if 0.575 <= min_dis_to_O < world.followers[0].r_F["r5"] * 1.5:
                        if self.min_dis_to_Os_old[idx] > min_dis_to_O:
                            R_O = - 20 * (self.min_dis_to_Os_old[idx] - min_dis_to_O)
                        else: R_O = 0
                        if self.is_close_to_Os[idx] and min_dis_to_O >= world.followers[0].r_F["r5"]:  # 復帰した場合正の報酬を与える
                            self.is_close_to_Os[idx] = False
                            R_O = 5
                    elif (min_dis_to_O < 0.575) and not self.is_close_to_Os[idx]: 
                        R_O = -7.5
                        self.is_close_to_Os[idx] = True
                    else: R_O = 0
                    self.R_O += R_O
                    self.min_dis_to_Os_old[idx] = min_dis_to_O
            else: self.R_O = 0
            
            # # 衝突に関する報酬
            is_col = self.funcs._check_col(L, world)
            if is_col and not self.is_col_old[0]: self.R_col = -1  # ぶつかった瞬間
            elif is_col and self.is_col_old[0]: self.R_col = 0  # ぶつかり続けている時(シミュレータの仕様でめり込む)
            elif not is_col and self.is_col_old[0]: self.R_col = 0  # 離れた時
            else: self.R_col = 0  # ずっとぶつかっていない時
            self.is_col_old[0] = is_col
            # self.R_col = 0
        else:  # 後ろのリーダーの二台目以降
           # リーダーがフォロワから離れすぎないための報酬
            self.min_dis_to_F = self.funcs._calc_min_dis_to_F(L, world)
            if self.min_dis_to_F > world.followers[0].r_L["r5d"] * 2.0:
                if self.min_dis_to_F > self.min_dis_to_F_old[1]:
                    self.R_L_close = - 0.05 * (self.min_dis_to_F - world.followers[0].r_L["r5d"])
                else: self.R_L_close = 0
            else: self.R_L_close = 0
            self.min_dis_to_F_old[1] = self.min_dis_to_F
            
            # リーダーが後ろ側に回り込むための報酬 
            L_dis_to_des = LA.norm(self.des - L.state.p_pos)  
            if L_dis_to_des < self.dis_to_des:
                self.R_back = - 0.05 * (self.dis_to_des - L_dis_to_des)
            else: self.R_back = 0
            
            # 衝突に関する報酬
            is_col = self.funcs._check_col(L, world)
            if is_col and not self.is_col_old[1]: self.R_col = -1  # ぶつかった瞬間
            elif is_col and self.is_col_old[1]: self.R_col = 0  # ぶつかり続けている時(シミュレータの仕様でめり込む)
            elif not is_col and self.is_col_old[1]: self.R_col = 0  # 離れた時
            else: self.R_col = 0  # ずっとぶつかっていない時
            self.is_col_old[1] = is_col 
        
        reward = self.R_F_far + self.R_g + self.R_div + self.R_L_close + self.R_back + self.R_O + self.R_col
        reward_list = np.array([self.R_F_far, self.R_g, self.R_div, self.R_L_close, self.R_back, self.R_O, self.R_col])
        
        return reward, reward_list
        
    def observation(self, L, world):
        i = int(L.name.replace('leader_', ''))
        if i < self.num_front_Ls + 1:  # 1台目のリーダー
            F_COM = self.funcs._calc_F_COM(world)
            self.COM_to_des = self.des - F_COM
            self.COM_to_des = self.funcs._coord_trans(self.COM_to_des)
            self.COM_to_des[0] /= self.max_dis_to_des  # 距離の正規化
            # 移動方向のベクトル
            self.COM_to_des_diff = - F_COM + self.F_COM_old
            self.COM_to_des_diff = self.funcs._coord_trans(self.COM_to_des_diff)
            self.COM_to_des_diff[0] /= self.max_moving_dis if self.COM_to_des_diff[0] <= self.max_moving_dis else self.COM_to_des_diff[0]
            
            COM_to_Os = []
            for idx, O in enumerate(world.obstacles):
                COM_to_O = O.state.p_pos - F_COM
                COM_to_O = self.funcs._coord_trans(COM_to_O)
                COM_to_O[0] /= self.max_dis_to_O if COM_to_O[0] < self.max_dis_to_O else COM_to_O[0]
                COM_to_Os.append(COM_to_O)
                # 移動方向のベクトル
                COM_to_O_diff = (O.state.p_pos - self.Os_old[idx]) - (F_COM - self.F_COM_old)
                COM_to_O_diff = self.funcs._coord_trans(COM_to_O_diff)
                COM_to_O_diff[0] /= self.max_moving_dis if COM_to_O_diff[0] <= self.max_moving_dis else COM_to_O_diff[0]
                COM_to_Os.append(COM_to_O_diff)
            # 1つのdequeを1stepで埋める
            self.COM_to_Os_deques[0].append(COM_to_Os)  # (num_Os, 4)
            
            # mask用の配列を用意
            self.mask_COM_to_O_list = []
            for idx in range(3 - self.num_Os):
                self.mask_COM_to_O_list.append(np.full((4, ), -10., np.float32))
            
            self.F_COM_old = F_COM
            self.Os_old = [O.state.p_pos for O in world.obstacles]
        
        L_to_Ls = []
        for other in world.agents:
            if L is other: continue 
            L_to_L = other.state.p_pos - L.state.p_pos
            L_to_L = self.funcs._coord_trans(L_to_L)
            L_to_L[0] /= self.max_dis_to_L  if L_to_L[0] < self.max_dis_to_L else L_to_L[0]  # 正規化
            L_to_Ls.append(L_to_L)
        self.L_to_Ls_deques[i].append(L_to_Ls)
        
        noise = 0.05 * (2 * np.random.rand() - 1)  # max5cmのホワイトノイズを加える
        L_to_Fs = []
        for F in world.followers:
            L_to_F = (F.state.p_pos - L.state.p_pos) + noise
            L_to_F = self.funcs._coord_trans(L_to_F)
            L_to_F[0] /= self.max_dis_to_F  if L_to_F[0] < self.max_dis_to_F else L_to_F[0]  # 正規化
            L_to_Fs.append(L_to_F)
        # to do: フォロワの並び替えの方法もう少し考える．    
        # self.L_to_Fs_deques[i].append(L_to_Fs)
        self.L_to_Fs_deques[i].append(np.sort(L_to_Fs, axis=0).tolist())
        
        # # Fの順番並べ替え用の配列を用意
        # if i < self.num_front_Ls + 1:  # 1台目のリーダー
        #     des_to_Fs = []
        #     for F in world.followers:
        #         des_to_F_dis = LA.norm(F.state.p_pos - self.des)
        #         des_to_Fs.append(des_to_F_dis)
        #     self.sort_idx = np.argsort(des_to_Fs).tolist()
        # self.L_to_Fs_deques[i].append(np.array(L_to_Fs)[self.sort_idx].tolist())
        
        # L_to_Os = []
        # for O in world.obstacles: 
        #     L_to_O = O.state.p_pos - L.state.p_pos
        #     L_to_O = self.funcs._coord_trans(L_to_O)
        #     L_to_O[0] /= self.max_dis_to_O if L_to_O[0] < self.max_dis_to_O else L_to_O[0]
        #     L_to_Os.append(L_to_O)
        # self.L_to_Os_deques[i].append(L_to_Os)
        
        obs = np.concatenate([self.COM_to_des] +[self.COM_to_des_diff] + [L.state.p_vel]
                            + self.L_to_Fs_deques[i][0] + self.L_to_Fs_deques[i][1]
                            + self.L_to_Ls_deques[i][0] + self.L_to_Ls_deques[i][1]
                            + self.COM_to_Os_deques[0][1] + self.mask_COM_to_O_list)

        # print(obs[-12:])
        return obs
    
    def check_done(self, L, world):
        # goalしたか否か
        # dis_to_des = self.funcs._calc_dis_to_des(world, self.des)
        # is_goal = self.funcs._check_goal(dis_to_des, self.rho_g)
        # 分裂したか否か
        # is_div = self.funcs._chech_div(world)
        # desまでの距離がmaxを超えていないか
        if self.dis_to_des > self.max_dis_to_des: is_exceed = True
        else: is_exceed = False
        # Fまでのmin距離がmaxを超えていないか
        if self.min_dis_to_F > self.max_dis_to_F: is_exceed_F = True
        else: is_exceed_F = False
        
        if self.is_goal or self.is_div or is_exceed or is_exceed_F: return True
        else: return False