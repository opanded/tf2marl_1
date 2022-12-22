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
import cv2

class Scenario(BaseScenario):   
    def __init__(self):
        # 学習時のエピソードのステップ数
        # リーダー数(前方，後方)，フォロワー数，障害物数の設定
        self.num_front_Ls = 0
        self.num_back_Ls = 2
        self.num_Ls = self.num_back_Ls + self.num_front_Ls
        self.num_Fs = 6
        self.num_Os = 0
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
        # add obstacles
        world.obstacles = [Obstacle() for i in range(self.num_Os)]
        for i, O in enumerate(world.obstacles):
            O.name = 'obstacle_%d' % i
            # O.size = 0.1
            O.collide = True
            O.movable = False
            O.color = np.array([0, 0.5, 0])
        self.max_dis_to_des = 12.5
        self.max_dis_to_agent = 7.5
        self.max_dis_to_O = 5. 
        self.arc_len_max = world.followers[0].r_F["r4"] * (self.num_Fs - 1) * 2
        # リーダー入れ替え用の配列を用意
        self.rand_idx = [i for i in range(self.num_Ls)]
        
        # make initial conditions
        self.reset_world(world)
        
        return world
    
    def reset_world(self, world):
        # goal到達時の閾値 
        self.rho_g = 1.0
        self.angle_des = np.radians(random.randint(0, 359))
        self.ini_dis_to_des = 5 + 2 * np.random.rand()
        
        if self.funcs._make_rand_sign() == 1 or -1:
            
            self.des = np.array([0, 0])
            
            self.F_pos = self.funcs._set_circle_F_pos(world, self.ini_dis_to_des, self.angle_des)  
            self.front_L_pos = self.funcs._set_front_L_pos(world, self.F_pos, self.num_front_Ls)
            self.back_L_pos = self.funcs._set_circle_back_L_pos(world, self.ini_dis_to_des, self.angle_des)
            self.O_pos = self.funcs._set_O_pos_st1(world, self.F_pos, self.des)
            
            # rotate F_pos
            rotate_angle = np.radians(random.randint(-180, 180))
            F_width = world.followers[0].r_F["r4"]
            self.F_pos = self.funcs._rotate_axis(self.F_pos, F_width, rotate_angle)
        else: 
            rand = 3 * np.random.rand()
            self.des = np.array([4.884644702315025455e+00, 4.619343504949231516e+00 + rand])
            self.F_pos = [np.array([3.206951570702668342e+00, 0.000000000000000000e+00]),
                                np.array([3.906951570702668519e+00, 0.000000000000000000e+00]),
                                np.array([4.606951570702667809e+00, 0.000000000000000000e+00]),
                                np.array([3.206951570702668342e+00, 6.999999999999999556e-01]),
                                np.array([3.906951570702668519e+00, 6.999999999999999556e-01]),
                                np.array([4.606951570702667809e+00, 6.999999999999999556e-01])]
            self.front_L_pos = self.funcs._set_front_L_pos(world, self.F_pos, self.num_front_Ls)
            self.back_L_pos = [np.array([3.411556323116248901e+00, -1.100000000000000089e+00]),
                                    np.array([4.898787262172401569e+00, -1.100000000000000089e+00])]
            self.O_pos = [np.array([4.045798136508846454e+00, 2.309671752474615758e+00 + rand])]
            
            rotate_angle = np.radians(180)
            F_width = world.followers[0].r_F["r4"]
            self.F_pos = self.funcs._rotate_axis(self.F_pos, F_width, rotate_angle)
            
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
        # set obstacles' random initial states   
        for i, O in enumerate(world.obstacles):
            O.state.p_pos = self.O_pos[i]
            O.state.p_vel = np.zeros(world.dim_p)

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
            if (self.max_dis_old - max_dis) > 1e-4: self.R_F_far = 5 * (self.max_dis_old - max_dis)
            else: self.R_F_far = 0
            self.max_dis_old = max_dis 
            self.R_F_far = np.clip(self.R_F_far, -0.1, 0.1)
            
            # 重心をゴールに近づける報酬
            self.dis_to_des = self.funcs._calc_dis_to_des(world, self.des)
            if self.dis_to_des > 2.0:
                self.R_g = 5 * (self.dis_to_des_old - self.dis_to_des)
            else:
                if (self.dis_to_des_old - self.dis_to_des) > 1e-4:
                    self.R_g = 5 * (self.dis_to_des_old - self.dis_to_des)  # denseな報酬
                else: self.R_g = -0.05
            self.R_g = np.clip(self.R_g, -0.1, 0.1)
            
            self.is_goal = self.funcs._check_goal(self.dis_to_des, self.rho_g)
            if self.is_goal: self.R_g = 1
            # 値の更新
            self.dis_to_des_old = self.dis_to_des
            
            # 分裂に関する報酬
            self.is_div = self.funcs._chech_div(world)
            if not self.is_div: self.R_div = 0
            else: self.R_div = -0.75
        
            # リーダーがフォロワから離れすぎないための報酬
            self.min_dis_to_F = self.funcs._calc_min_dis_to_F(L, world)
            if self.min_dis_to_F > world.followers[0].r_L["r5d"] * 2.0:
                if self.min_dis_to_F > self.min_dis_to_F_old[0]:
                    self.R_L_close = - 0.05 * (self.min_dis_to_F - world.followers[0].r_L["r5d"])
                else: self.R_L_close = 0.01
            else: self.R_L_close = 0
            self.min_dis_to_F_old[0] = self.min_dis_to_F
            self.R_L_close = np.clip(self.R_L_close, -0.2, 0.1)
            
            # リーダーが後ろ側に回り込むための報酬
            L_dis_to_des = LA.norm(self.des - L.state.p_pos)
            if L_dis_to_des < self.dis_to_des:
                self.R_back = - 0.05 * (self.dis_to_des - L_dis_to_des)
            else: self.R_back = 0
            self.R_back = np.clip(self.R_back, -0.05, 0.05)
            
            # 障害物に関する報酬
            self.R_O = 0
            
            # 衝突に関する報酬
            is_col = self.funcs._check_col(L, world)
            if is_col and not self.is_col_old[0]: self.R_col = -0.75  # ぶつかった瞬間
            elif is_col and self.is_col_old[0]: self.R_col = 0  # ぶつかり続けている時(シミュレータの仕様でめり込む)
            elif not is_col and self.is_col_old[0]: self.R_col = 0  # 離れた時
            else: self.R_col = 0  # ずっとぶつかっていない時
            self.is_col_old[0] = is_col
        else:  # 後ろのリーダーの二台目以降
            # リーダーがフォロワから離れすぎないための報酬更新
            self.min_dis_to_F = self.funcs._calc_min_dis_to_F(L, world)
            if self.min_dis_to_F > world.followers[0].r_L["r5d"] * 2.0:
                if self.min_dis_to_F > self.min_dis_to_F_old[1]:
                    self.R_L_close = - 0.05 * (self.min_dis_to_F - world.followers[0].r_L["r5d"])
                else: self.R_L_close = 0.01
            else: self.R_L_close = 0
            self.min_dis_to_F_old[1] = self.min_dis_to_F
            self.R_L_close = np.clip(self.R_L_close, -0.2, 0.1)
            
            # リーダーが後ろ側に回り込むための報酬 
            L_dis_to_des = LA.norm(self.des - L.state.p_pos)  
            if L_dis_to_des < self.dis_to_des:
                self.R_back = - 0.05 * (self.dis_to_des - L_dis_to_des)
            else: self.R_back = 0
            self.R_back = np.clip(self.R_back, -0.05, 0.05)
            
            # 衝突に関する報酬
            is_col = self.funcs._check_col(L, world)
            if is_col and not self.is_col_old[1]: self.R_col = -0.75  # ぶつかった瞬間
            elif is_col and self.is_col_old[1]: self.R_col = 0  # ぶつかり続けている時(シミュレータの仕様でめり込む)
            elif not is_col and self.is_col_old[1]: self.R_col = 0  # 離れた時
            else: self.R_col = 0  # ずっとぶつかっていない時
            self.is_col_old[1] = is_col
        
        reward = self.R_F_far + self.R_g + self.R_div + self.R_L_close + self.R_back + self.R_O + self.R_col
        reward_list = np.array([self.R_F_far, self.R_g, self.R_div, self.R_L_close, self.R_back, self.R_O, self.R_col])
        
        return reward, reward_list
        
    def observation(self, L, world):
        L_idx = int(L.name.replace('leader_', ''))
        if L_idx < self.num_front_Ls + 1:  # 1台目のリーダー
            self.F_COM = self.funcs._calc_F_COM(world)
            self.COM_to_des = self.des - self.F_COM
            self.COM_to_des = self.funcs._coord_trans(self.COM_to_des)
            self.COM_to_des[0] /= self.max_dis_to_des  # 距離の正規化
            # 移動方向のベクトル
            self.COM_to_des_diff = - self.F_COM + self.F_COM_old
            self.COM_to_des_diff = self.funcs._coord_trans(self.COM_to_des_diff)
            self.COM_to_des_diff[0] /= self.max_moving_dis if self.COM_to_des_diff[0] <= self.max_moving_dis else self.COM_to_des_diff[0]
            
                        
            COM_to_Os = []
            for idx, O in enumerate(world.obstacles):
                COM_to_O = O.state.p_pos - self.F_COM
                COM_to_O = self.funcs._coord_trans(COM_to_O)
                COM_to_O[0] /= self.max_dis_to_O if COM_to_O[0] < self.max_dis_to_O else COM_to_O[0]
                COM_to_Os.append(COM_to_O)
                # 移動方向のベクトル
                COM_to_O_diff = (O.state.p_pos - self.Os_old[idx]) - (self.F_COM - self.F_COM_old)
                COM_to_O_diff = self.funcs._coord_trans(COM_to_O_diff)
                COM_to_O_diff[0] /= self.max_moving_dis if COM_to_O_diff[0] <= self.max_moving_dis else COM_to_O_diff[0]
                COM_to_Os.append(COM_to_O_diff)
            self.COM_to_Os_deques[0].append(COM_to_Os)  # (num_Os, 4)
            
            # mask用の配列を用意
            self.mask_COM_to_O_list = []
            for idx in range(3 - self.num_Os):
                self.mask_COM_to_O_list.append(np.full((4, ), -10., np.float32))
            
            self.F_COM_old = self.F_COM
            self.Os_old = [O.state.p_pos for O in world.obstacles]
            
            # 群れの外接多角形の長さを取得
            self.F_pts = [F.state.p_pos for F in world.followers]
            hull = cv2.convexHull(np.array(self.F_pts, dtype=np.float32))
            self.hull_len = cv2.arcLength(hull, True)
            self.hull_len /= self.arc_len_max if self.hull_len <= self.arc_len_max else self.hull_len
            # 外接矩形の4点を取得
            rect = cv2.minAreaRect(hull)
            world.box = cv2.boxPoints(rect)  # renderingに表示させるためにworldの変数とする
        
        L_to_Ls = []
        for other in world.agents:
            if L is other: continue 
            L_to_L = other.state.p_pos - L.state.p_pos
            L_to_L = self.funcs._coord_trans(L_to_L)
            L_to_L[0] /= self.max_dis_to_agent  if L_to_L[0] < self.max_dis_to_agent else L_to_L[0]  # 正規化
            L_to_Ls.append(L_to_L)
        self.L_to_Ls_deques[L_idx].append(L_to_Ls)
        
        # # 全てのフォロワの座標を入力とする
        # L_to_Fs = []
        # for F in world.followers:
        #     noise = 0.1 * (2 * np.random.rand() - 1)  # max10cmのホワイトノイズ
        #     L_to_F = F.state.p_pos - L.state.p_pos + noise
        #     L_to_F = self.funcs._coord_trans(L_to_F)
        #     L_to_F[0] /= self.max_dis_to_agent  if L_to_F[0] < self.max_dis_to_agent else L_to_F[0]  # 正規化
        #     L_to_Fs.append(L_to_F)
        # L_to_Fs = np.array(L_to_Fs)
        # L_to_Fs = L_to_Fs[np.argsort(L_to_Fs[:, 0])]
        # self.L_to_Fs_deques[L_idx].append(L_to_Fs.tolist())
        
        # 外接円の4点を入力とし，近い順に並び替える
        L_to_circ_vecs = []
        L_to_COM = self.F_COM - L.state.p_pos
        COM_to_far_F = np.array([0, 0])
        for F in world.followers:
            COM_to_F = F.state.p_pos - self.F_COM
            if LA.norm(COM_to_F) >= LA.norm(COM_to_far_F):
                COM_to_far_F =  COM_to_F
        for i in range(4):
            R = np.array([[np.cos(i * 90), -np.sin(i * 90)],
                          [np.sin(i * 90),  np.cos(i * 90)]])
            COM_to_circ_vec = np.dot(R, COM_to_far_F)
            L_to_circ_vec = L_to_COM + COM_to_circ_vec
            L_to_circ_vec = self.funcs._coord_trans(L_to_circ_vec)
            L_to_circ_vec[0] /= self.max_dis_to_agent if L_to_circ_vec[0] < self.max_dis_to_agent else L_to_circ_vec[0]
            L_to_circ_vecs.append(L_to_circ_vec)
        L_to_circ_vecs = np.array(L_to_circ_vecs)
        L_to_circ_vecs = L_to_circ_vecs[np.argsort(L_to_circ_vecs[:, 0])]
        self.L_to_Fs_deques[L_idx].append(L_to_circ_vecs.tolist())
        
        # # 外接矩形の4点を入力とし，近い順に並び替える
        # L_to_rec_vecs = []
        # for point in world.box:
        #     L_to_rec_vec = point - L.state.p_pos
        #     L_to_rec_vec = self.funcs._coord_trans(L_to_rec_vec)
        #     L_to_rec_vec[0] /= self.max_dis_to_agent if L_to_rec_vec[0] < self.max_dis_to_agent else L_to_rec_vec[0]
        #     L_to_rec_vecs.append(L_to_rec_vec)
        # L_to_rec_vecs = np.array(L_to_rec_vecs)
        # L_to_rec_vecs = L_to_rec_vecs[np.argsort(L_to_rec_vecs[:, 0])]
        # self.L_to_Fs_deques[L_idx].append(L_to_rec_vecs.tolist())
        
        # 1番近い物体までの相対ベクトル
        L_to_min_obj = np.array([100, 100])
        for obj in world.entities:
            if L is obj: continue
            L_to_obj = obj.state.p_pos - L.state.p_pos
            L_to_obj = self.funcs._coord_trans(L_to_obj)
            if L_to_obj[0] < L_to_min_obj[0]:
                L_to_min_obj = L_to_obj
        L_to_min_obj[0] /= self.max_dis_to_agent if L_to_obj[0] < self.max_dis_to_agent else L_to_min_obj[0]
        
        
        obs = np.concatenate([self.COM_to_des] + [self.COM_to_des_diff] 
                            + [L.state.p_vel] + [[self.hull_len]]
                            + self.L_to_Fs_deques[L_idx][1] + self.L_to_Ls_deques[L_idx][1] + [L_to_min_obj]
                            + self.COM_to_Os_deques[0][1] + self.mask_COM_to_O_list)
        
        # obs = np.concatenate([self.COM_to_des] 
        #                     + [L.state.p_vel] + [[self.hull_len]]
        #                     + self.L_to_Fs_deques[L_idx][1] + self.L_to_Ls_deques[L_idx][1] + [L_to_min_obj]
        #                     + self.COM_to_Os_deques[0][1] + self.mask_COM_to_O_list)
        
        return obs
    
    def check_done(self, L, world):
        # desまでの距離がmaxを超えていないか
        if self.dis_to_des > self.max_dis_to_des: 
            is_exceed = True
        else: is_exceed = False
        # Fまでのmin距離がmaxを超えていないか
        if self.min_dis_to_F > self.max_dis_to_agent: is_exceed_F = True
        else: is_exceed_F = False
        
        if self.is_goal:
            return True, "goal"
        elif self.is_div:
            return True, "divide"
        elif is_exceed or is_exceed_F: 
            return True, "exceed"
        else: 
            return False, "continue"