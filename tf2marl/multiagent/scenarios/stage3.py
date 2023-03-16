import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
from multiagent.core import World, Agent, Follower, Obstacle, Obstacle_cross
from multiagent.scenario import BaseScenario
from multiagent.scenarios.base_funcs import Basefuncs
from numpy import linalg as LA
import copy
import random
import cv2

class Scenario(BaseScenario):   
    def __init__(self):
        # リーダー数，フォロワー数，障害物数の設定
        self.num_Ls = 2
        self.num_Fs = 6
        self.num_Os = random.choice([1, 2, 3])
        self.funcs = Basefuncs()
        self.is_display = False
        self.is_evaluate = False
    
    def make_world(self):
        world = World()
        # add Leaders
        world.agents = [Agent() for i in range(self.num_Ls)]
        for i, L in enumerate(world.agents):
            L.name = 'leader_%d' % i
            L.color = np.array([1, 0, 0])
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
        self.max_moving_dis = 2 * world.agents[0].max_speed * world.dt
        # リーダー入れ替え用の配列を用意
        self.rand_idx = [i for i in range(self.num_Ls)]
        # make initial conditions
        self.reset_world(world)
        return world
    
    def reset_world(self, world):
        # set goal configration
        if (not self.is_display) and (not self.is_evaluate): self.rho_g = 1.1
        else: self.rho_g = 1.25
        self.des = np.array([0, 8])
        
        # フォロワの数をエピソード毎に変化させる
        # if (not self.is_display) or self.is_evaluate:  # 学習時 or 評価時 
        #   self.num_Fs = random.choice([4, 5, 6, 7, 8, 9])
        # else: self.num_Fs = 9
        # world.followers = [Follower() for i in range(self.num_Fs)]
        
        # set initial object position
        self.F_pos = self.funcs._set_F_pos(world) 
        self.L_pos = self.funcs._set_L_pos(world, self.F_pos, self.num_Ls)
        
        if (not self.is_display) and (not self.is_evaluate):  # 学習時 
            if np.random.rand() <= 0.5: flag = 0
            else: flag = 1
        else: flag = 1
        # set obstacle configuration
        if flag == 0:  # crossing
            if (not self.is_display) or self.is_evaluate:  # 学習時 or 評価時 
                self.num_Os = random.choice([1, 2])
            else: self.num_Os = 2  # rendering時
            world.obstacles = [Obstacle_cross() for i in range(self.num_Os)]
            self.O_pos = self.funcs._set_crossing_O_pos(world, self.F_pos, self.des)
        else:  # remain
            if (not self.is_display) or self.is_evaluate:  # 学習時 or 評価時  
                self.num_Os = random.choice([2, 3])
            else: self.num_Os = 3  # rendering時
            world.obstacles = [Obstacle() for i in range(self.num_Os)]
            self.O_pos = self.funcs._set_O_pos(world, self.F_pos, self.des)
            idx_list =[i for i in range(self.num_Os)]
            idx = random.choices(idx_list, k=self.num_Os)
        # set leader's random initial states
        for i, L in enumerate(world.agents):
            L.state.p_pos = self.L_pos[i]
            L.state.p_vel = np.zeros(world.dim_p)
        # set follower's random initial states
        for i, F in enumerate(world.followers):
            F.name = 'follower_%d' % i
            F.state.p_pos = self.F_pos[i]
            F.state.p_vel = np.zeros(world.dim_p)
            
        O_to_COMs = []; F_COM = self.funcs._calc_F_COM(world)
        for i, O in enumerate(world.obstacles):
            O.name = 'obstacle_%d' % i
            O.state.p_pos = self.O_pos[i]
            O_to_COMs.append(LA.norm(O.state.p_pos - F_COM))
        # set obstacle's random initial states        
        for i, O in enumerate(world.obstacles):
            if flag == 0:  # crossing
                close_id = np.argsort(O_to_COMs)
                # O.start_step = close_id[i] * 125
                O.start_step = close_id[i] * 0
            else:  # remain
                if i in idx: 
                    O.have_vel = True
                    O.init_pos = copy.deepcopy(self.O_pos[i])
                    x_sign = 1 if np.random.rand() <= 0.5 else -1
                    y_sign = 1 if np.random.rand() <= 0.5 else -1
                    vel = 0.2
                    O.state.p_vel = np.array([vel * x_sign, vel * y_sign])
        # for reward
        # ゴールから一番遠いフォロワの距離
        self.max_dis_old = 0; self.max_dis_idx_old = 0
        # 重心からゴールまでの距離
        self.dis_to_des_old = self.funcs._calc_dis_to_des(world, self.des)
        # 各リーダーから一番近いフォロワまでの距離
        self.min_dis_to_F_old = [self.funcs._calc_min_dis_to_F(L, world) for L in world.agents]
        # 各リーダーから目的地までの距離
        self.L_dis_to_des_old = [LA.norm(self.des - L.state.p_pos) for L in world.agents]
        # 各障害物用の変数
        self.is_close_to_Os = []; self.min_dis_to_Os_old = []
        for _ in range(self.num_Os):
            self.is_close_to_Os.append(False)
            self.min_dis_to_Os_old.append(self.funcs._calc_Fs_min_dis_to_Os(world))
        # 衝突判定用の変数
        self.is_col_old = [False for _ in range(self.num_Ls)]
        # for observation
        # フォロワの重心位置
        self.F_COM_old = F_COM
        # 各障害物の位置
        self.Os_old = [O.state.p_pos for O in world.obstacles]
            
        return self.des, self.rho_g, self.funcs._calc_F_COM
    
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
            if (self.dis_to_des_old - self.dis_to_des) > 1e-4:
                self.R_g = 7.5 * (self.dis_to_des_old - self.dis_to_des)
            elif -1e-4 < (self.dis_to_des_old - self.dis_to_des) <= 1e-4: 
                self.R_g = -0.01  # スタックを防止するために一箇所で止まっている場合マイナスの報酬を入れる
            else: self.R_g = -0.05  # 離れた場合大きなマイナス報酬を入れる
            self.R_g = np.clip(self.R_g, -0.1, 0.1)
            # ゴールした場合
            self.is_goal = self.funcs._check_goal(self.dis_to_des, self.rho_g)
            if self.is_goal: self.R_g = 1
            self.dis_to_des_old = self.dis_to_des
            # 分裂に関する報酬
            self.is_div = self.funcs._chech_div(world)
            if not self.is_div: self.R_div = 0
            else: self.R_div = -0.75
            # 障害物に近づきすぎないための報酬
            if world.obstacles:
                self.R_O = 0
                min_dis_to_Os = self.funcs._calc_Fs_min_dis_to_Os(world)
                for idx, min_dis_to_O in enumerate(min_dis_to_Os):
                    if 0.575 <= min_dis_to_O < world.followers[0].r_F["r5"] * 1.5:
                        if self.min_dis_to_Os_old[idx] > min_dis_to_O:
                            R_O = - 20 * (self.min_dis_to_Os_old[idx] - min_dis_to_O)
                        else: R_O = 0
                        if self.is_close_to_Os[idx] and min_dis_to_O >= world.followers[0].r_F["r5"]:  # 復帰した場合正の報酬を与える
                            self.is_close_to_Os[idx] = False
                            R_O = 0.5
                    elif (min_dis_to_O < 0.575) and not self.is_close_to_Os[idx]: 
                        R_O = -0.75
                        self.is_close_to_Os[idx] = True
                    else: R_O = 0
                    self.R_O += R_O
                    self.min_dis_to_Os_old[idx] = min_dis_to_O
                self.R_O = np.clip(self.R_O, -0.75, 0.5)
            else: self.R_O = 0
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
            if L_dis_to_des < self.L_dis_to_des_old[L_idx]:
                self.R_back = - 0.05 * (self.dis_to_des - L_dis_to_des)
            else: self.R_back = 0
        else: self.R_back = 0
        self.L_dis_to_des_old[L_idx] = L_dis_to_des
        self.R_back = np.clip(self.R_back, -0.05, 0.05)
        # 衝突に関する報酬
        min_dis = self.funcs._calc_min_dis(L, world)
        if min_dis > 0:
            is_col = False
            if min_dis < 0.4:  # 40cmより近い場合負の報酬を入れる
                self.R_col = - 1 / (5 + min_dis)
            else: self.R_col = 0
        else:
            is_col = True
            if is_col and not self.is_col_old[L_idx]: self.R_col = -1.0  # ぶつかった瞬間
            else: self.R_col = 0  # ぶつかり続けている時(シミュレータの仕様でめり込む)
        self.is_col_old[L_idx] = is_col
        
        reward = self.R_F_far + self.R_g + self.R_div + self.R_L_close + self.R_back + self.R_O + self.R_col
        reward_list = np.array([self.R_F_far, self.R_g, self.R_div, self.R_L_close, self.R_back, self.R_O, self.R_col])
        
        return reward, reward_list
        
    def observation(self, L, world):
        L_idx = int(L.name.replace('leader_', ''))  # リーダーのID
        if L_idx < 1:  # 1台目のリーダー
            self.F_COM = self.funcs._calc_F_COM(world)
            self.COM_to_des = self.des - self.F_COM
            self.COM_to_des = self.funcs._coord_trans(self.COM_to_des)
            self.COM_to_des[0] /= self.max_dis_to_des if self.COM_to_des[0] <= self.max_dis_to_des else self.COM_to_des[0]
                       
            COM_to_Os = []; COM_to_Os_diff = []
            for idx, O in enumerate(world.obstacles):
                COM_to_O_dis = LA.norm(O.state.p_pos - self.F_COM)
                # 物体表面までのベクトルを計算
                COM_to_O = (O.state.p_pos - self.F_COM) / COM_to_O_dis * (COM_to_O_dis - O.size)
                COM_to_O = self.funcs._coord_trans(COM_to_O)
                COM_to_O[0] /= self.max_dis_to_O if COM_to_O[0] < self.max_dis_to_O else COM_to_O[0]
                COM_to_Os.append(COM_to_O)
                # 相対速度ベクトル
                COM_to_O_diff = (O.state.p_pos - self.Os_old[idx]) - (self.F_COM - self.F_COM_old)
                COM_to_O_diff = self.funcs._coord_trans(COM_to_O_diff)
                COM_to_O_diff[0] /= self.max_moving_dis if COM_to_O_diff[0] <= self.max_moving_dis else COM_to_O_diff[0]
                COM_to_Os_diff.append(COM_to_O_diff)
            # 障害物を遠い順に並び替える
            if self.num_Os != 0:
                COM_to_Os = np.array(COM_to_Os); COM_to_Os_diff = np.array(COM_to_Os_diff)
                sort_idx = np.argsort(COM_to_Os[:, 0])[::-1]
                COM_to_Os = COM_to_Os[sort_idx]
                COM_to_Os_diff = COM_to_Os_diff[sort_idx]
            # 最終的な障害物に関するobservation
            self.COM_to_Os_info = []
            for O_vec, O_diff in zip(COM_to_Os, COM_to_Os_diff):
                self.COM_to_Os_info.append(O_vec)
                self.COM_to_Os_info.append(O_diff)
            # mask用の配列を用意
            self.mask_COM_to_Os = [np.full((4, ), -10., np.float32) for _ in range(3 - self.num_Os)]
            # old変数の更新
            self.F_COM_old = self.F_COM
            self.Os_old = [copy.deepcopy(O.state.p_pos) for O in world.obstacles]
            # 外接矩形の4点を取得
            self.F_pts = [F.state.p_pos for F in world.followers]
            hull = cv2.convexHull(np.array(self.F_pts, dtype=np.float32))
            rect = cv2.minAreaRect(hull)
            world.box = cv2.boxPoints(rect)  # renderingに表示させるためにworldの変数とする
            
        # リーダーから見た他のリーダーの座標
        L_to_Ls = []
        for other in world.agents:
            if L is other: continue 
            L_to_L = other.state.p_pos - L.state.p_pos
            L_to_L = self.funcs._coord_trans(L_to_L)
            L_to_L[0] /= self.max_dis_to_agent  if L_to_L[0] < self.max_dis_to_agent else L_to_L[0]
            L_to_Ls.append(L_to_L)
        # 外接矩形の4点を入力とし，近い順に並び替える
        L_to_rec_vecs = []
        for point in world.box:
            L_to_rec_vec = point - L.state.p_pos
            L_to_rec_vec = self.funcs._coord_trans(L_to_rec_vec)
            L_to_rec_vec[0] /= self.max_dis_to_agent if L_to_rec_vec[0] < self.max_dis_to_agent else L_to_rec_vec[0]
            L_to_rec_vecs.append(L_to_rec_vec)
        L_to_rec_vecs = np.array(L_to_rec_vecs)
        L_to_rec_vecs = L_to_rec_vecs[np.argsort(L_to_rec_vecs[:, 0])]  # リーダーからの距離に応じてソート
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

        obs = np.concatenate([self.COM_to_des] + [L.state.p_vel]
                            + L_to_Fs + L_to_Ls + [L_to_min_obj]
                            + self.COM_to_Os_info + self.mask_COM_to_Os)

        return obs
    
    def check_done(self, L, world):
        # desまでの距離がmaxを超えていないか
        if self.dis_to_des > self.max_dis_to_des: 
            is_exceed = True
        else: is_exceed = False
        # Fまでのmin距離がmaxを超えていないか
        if self.min_dis_to_F > self.max_dis_to_agent: is_exceed_F = True
        else: is_exceed_F = False
        
        # 学習時と評価時で終了条件を変える
        if (not self.is_display) and (not self.is_evaluate):
            if self.is_goal:
                return True, "goal"
            elif self.is_div:
                return True, "divide"
            elif any(self.is_col_old):
                return True, "collide"
            elif is_exceed or is_exceed_F: 
                return True, "exceed"
            else: 
                return False, "continue"
        else:
            if self.is_goal:
                return True, "goal"
            elif self.is_div:
                return True, "divide"
            elif any(self.is_col_old):
                return True, "collide"
            else: 
                return False, "continue"