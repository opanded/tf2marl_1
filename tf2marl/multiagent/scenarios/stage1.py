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
        # リーダー数，フォロワー数，障害物数の設定
        self.num_Ls = 2
        self.num_Fs = 6
        self.num_Os = 0  # 障害物なし環境にて学習させる
        self.funcs = Basefuncs()
    
    def make_world(self):
        world = World()
        # add Leaders
        world.agents = [Agent() for i in range(self.num_Ls)]
        for i, L in enumerate(world.agents):
            L.name = 'leader_%d' % i
            if i == 0: L.color = np.array([1, 0, 0])
            else: L.color = np.array([1, 0.5, 0])
        # add Followers
        world.followers = [Follower() for i in range(self.num_Fs)]
        for i, F in enumerate(world.followers):
            F.name = 'follower_%d' % i
            F.collide = True
            F.movable = False
            F.color = np.array([0, 0, 1])
        # set max distance
        self.max_dis_to_des = 12.5
        self.max_dis_to_agent = 7.5
        self.max_dis_to_O = 5. 
        self.arc_len_max = world.followers[0].r_F["r4"] * (self.num_Fs - 1) * 2
        self.max_moving_dis = 2 * world.agents[0].max_speed * world.dt
        # リーダー入れ替え用の配列を用意
        self.rand_idx = [i for i in range(self.num_Ls)]
        # make initial conditions
        self.reset_world(world)
        return world
    
    def reset_world(self, world):
        # # フォロワの数をエピソード毎に変化させる
        # self.num_Fs = random.choice([4, 5, 6, 7, 8, 9])
        # world.followers = [Follower() for i in range(self.num_Fs)]
        # coh = 1 + np.random.rand()
        # for F in world.followers:
        #     F.k_FF_coh = coh
        
        # set goal configration
        self.rho_g = 1.0
        self.des = np.array([0, 0])
        self.ini_dis_to_des = 5 + 2 * np.random.rand()    
        self.angle_des = np.radians(random.randint(0, 359))
        # set initial object position
        self.F_pos = self.funcs._set_F_pos_st1(world, self.ini_dis_to_des, self.angle_des)  
        self.L_pos = self.funcs._set_L_pos_st1(world, self.ini_dis_to_des, self.angle_des)
        # rotate F_pos
        rotate_angle = np.radians(random.randint(-180, 180))
        F_width = world.followers[0].r_F["r4"]
        self.F_pos = self.funcs._rotate_axis(self.F_pos, F_width, rotate_angle)
        # 一定の確率でリーダーのポジションを入れ替える
        L_i = random.choice(self.rand_idx); L_j = random.choice(self.rand_idx)
        if L_i != L_j:
           tmp = self.L_pos[L_i]
           self.L_pos[L_i] = self.L_pos[L_j]
           self.L_pos[L_j] = tmp
        # 初期位置を記録するための配列
        pos_dict = {"follower": copy.deepcopy(self.F_pos),"leader": copy.deepcopy(self.L_pos)
                    ,"dest": [copy.deepcopy(self.des)]}
        # set leader's random initial states
        for i, L in enumerate(world.agents):
            L.state.p_pos = self.L_pos[i]
            L.state.p_vel = np.zeros(world.dim_p)
        # set follower's random initial states
        for i, F in enumerate(world.followers):
            F.name = 'follower_%d' % i
            F.state.p_pos = self.F_pos[i]
            F.state.p_vel = np.zeros(world.dim_p)
        # For reward
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
            self.min_dis_to_Os_old.append(self.funcs._calc_Fs_min_dis_to_Os(world))
        # 衝突判定用の変数
        self.is_col_old = [False for _ in range(self.num_Ls)]
        # For observation
        # フォロワの重心位置
        self.F_COM_old = self.funcs._calc_F_COM(world)
        # 各障害物の位置
        self.Os_old = [O.state.p_pos for O in world.obstacles]
    
        return self.des, self.rho_g, self.funcs._calc_F_COM, pos_dict
    
    def reward(self, L, world):
        L_idx = int(L.name.replace('leader_', ''))
        if L_idx < 1:  # 1台目のリーダー
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
            self.dis_to_des_old = self.dis_to_des
            # 分裂に関する報酬
            self.is_div = self.funcs._chech_div(world)
            if not self.is_div: self.R_div = 0
            else: self.R_div = -0.75
            # 障害物に関する報酬
            self.R_O = 0
            
        # リーダーがフォロワから離れすぎないための報酬
        self.min_dis_to_F = self.funcs._calc_min_dis_to_F(L, world)
        if self.min_dis_to_F > world.followers[0].r_L["r5d"] * 2.0:
            if self.min_dis_to_F > self.min_dis_to_F_old[L_idx]:
                self.R_L_close = - 0.05 * (self.min_dis_to_F - world.followers[0].r_L["r5d"])
            else: self.R_L_close = 0.01
        else: self.R_L_close = 0
        self.min_dis_to_F_old[L_idx] = self.min_dis_to_F
        self.R_L_close = np.clip(self.R_L_close, -0.2, 0.1)
        # リーダーが後ろ側に回り込むための報酬 
        L_dis_to_des = LA.norm(self.des - L.state.p_pos)  
        if L_dis_to_des < self.dis_to_des:
            self.R_back = - 0.05 * (self.dis_to_des - L_dis_to_des)
        else: self.R_back = 0
        self.R_back = np.clip(self.R_back, -0.05, 0.05)        
        # 衝突に関する報酬
        is_col = self.funcs._check_col(L, world)
        if is_col and not self.is_col_old[L_idx]: self.R_col = -0.75  # ぶつかった瞬間
        elif is_col and self.is_col_old[L_idx]: self.R_col = 0  # ぶつかり続けている時(シミュレータの仕様でめり込む)
        elif not is_col and self.is_col_old[L_idx]: self.R_col = 0  # 離れた時
        else: self.R_col = 0  # ずっとぶつかっていない時
        self.is_col_old[L_idx] = is_col
        
        reward = self.R_F_far + self.R_g + self.R_div + self.R_L_close + self.R_back + self.R_O + self.R_col
        reward_list = np.array([self.R_F_far, self.R_g, self.R_div, self.R_L_close, self.R_back, self.R_O, self.R_col])
        
        return reward, reward_list
        
    def observation(self, L, world):
        L_idx = int(L.name.replace('leader_', ''))
        if L_idx < 1:  # 1台目のリーダー
            self.F_COM = self.funcs._calc_F_COM(world)
            self.COM_to_des = self.des - self.F_COM
            self.COM_to_des = self.funcs._coord_trans(self.COM_to_des)
            self.COM_to_des[0] /= self.max_dis_to_des  # 距離の正規化
            # 移動方向のベクトル
            self.COM_to_des_diff = - self.F_COM + self.F_COM_old
            self.COM_to_des_diff = self.funcs._coord_trans(self.COM_to_des_diff)
            self.COM_to_des_diff[0] /= self.max_moving_dis if self.COM_to_des_diff[0] <= self.max_moving_dis else self.COM_to_des_diff[0]
            # 障害物用の配列はst1においては空
            self.COM_to_Os = []
            # mask用の配列を用意
            self.mask_COM_to_Os = [np.full((4, ), -10., np.float32) for _ in range(3 - self.num_Os)]
            # 重心位置と障害物の位置を更新
            self.F_COM_old = self.F_COM
            self.Os_old = [O.state.p_pos for O in world.obstacles]
            
            # 群れの外接多角形の長さを取得
            self.F_pts = [F.state.p_pos for F in world.followers]
            hull = cv2.convexHull(np.array(self.F_pts, dtype=np.float32))
            # self.hull_len = cv2.arcLength(hull, True)
            # self.hull_len /= self.arc_len_max if self.hull_len <= self.arc_len_max else self.hull_len
            # 外接矩形の4点を取得
            rect = cv2.minAreaRect(hull)
            world.box = cv2.boxPoints(rect)  # renderingに表示させるためにworldの変数とする
        
        # リーダーから見たゴールの座標
        L_to_des = self.des - L.state.p_pos
        L_to_des = self.funcs._coord_trans(L_to_des)
        L_to_des[0] /= self.max_dis_to_des if L_to_des[0] <= self.max_dis_to_des else L_to_des[0]
        # リーダーから見た他のリーダーの座標
        L_to_Ls = []
        for other in world.agents:
            if L is other: continue 
            L_to_L = other.state.p_pos - L.state.p_pos
            L_to_L = self.funcs._coord_trans(L_to_L)
            L_to_L[0] /= self.max_dis_to_agent  if L_to_L[0] < self.max_dis_to_agent else L_to_L[0]  # 正規化
            L_to_Ls.append(L_to_L)
        
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
        # L_to_Fs = L_to_Fs.tolist()
        
        # 外接矩形の4点を入力とし，近い順に並び替える
        L_to_rec_vecs = []
        for point in world.box:
            L_to_rec_vec = point - L.state.p_pos
            L_to_rec_vec = self.funcs._coord_trans(L_to_rec_vec)
            L_to_rec_vec[0] /= self.max_dis_to_agent if L_to_rec_vec[0] < self.max_dis_to_agent else L_to_rec_vec[0]
            L_to_rec_vecs.append(L_to_rec_vec)
        L_to_rec_vecs = np.array(L_to_rec_vecs)
        L_to_rec_vecs = L_to_rec_vecs[np.argsort(L_to_rec_vecs[:, 0])]
        L_to_Fs = L_to_rec_vecs.tolist()
        
        # 1番近い物体までの相対ベクトル
        L_to_min_obj = np.array([100, 100])
        for obj in world.entities:
            if L is obj: continue
            L_to_obj = obj.state.p_pos - L.state.p_pos
            L_to_obj = self.funcs._coord_trans(L_to_obj)
            if L_to_obj[0] < L_to_min_obj[0]:
                L_to_min_obj = L_to_obj
        L_to_min_obj[0] /= self.max_dis_to_agent if L_to_obj[0] < self.max_dis_to_agent else L_to_min_obj[0]
        
        obs = np.concatenate([self.COM_to_des] + [L_to_des] + [L.state.p_vel]
                            + L_to_Fs + L_to_Ls + [L_to_min_obj]
                            + self.COM_to_Os + self.mask_COM_to_Os)

        # obs = np.concatenate([self.COM_to_des] + [L.state.p_vel]
        #                     + L_to_Fs + L_to_Ls + [L_to_min_obj]
        #                     + self.COM_to_Os + self.mask_COM_to_Os)

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