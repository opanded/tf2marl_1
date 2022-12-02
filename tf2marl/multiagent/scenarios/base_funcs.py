import numpy as np
from numpy import linalg as LA
from sklearn.cluster import DBSCAN
###########################################  For reset_world ###########################################
class Basefuncs():

    def _make_rand_sign(self):
        if np.random.rand() <= 0.5: return 1
        else: return -1


    def _set_F_pos(self, world):
        F_pos = []
        # 3台ずつ列状に並べる
        n = 3
        m = -(-len(world.followers) // n) # m: 列数，演算子を用いて切り上げをしている
        F_ref_coord = np.array([5 * (2 * np.random.rand() - 1), (2 * np.random.rand() - 1)])
        F_width = world.followers[0].r_F["r4"]
        for i in range(m): 
            for j in range(n):
                F_next_coord = np.array([F_ref_coord[0] + j * F_width, i * F_width]) 
                F_pos.append(F_next_coord)

        return F_pos    

    def _set_circle_F_pos(self, world, ini_dis_to_des, angle_des):
        fol_width = world.followers[0].r_F["r4"]
        fol_pos = [ini_dis_to_des * np.array([np.cos(angle_des), np.sin(angle_des)]),
                    (ini_dis_to_des + fol_width) * np.array([np.cos(angle_des), np.sin(angle_des)])]
        
        for i in range(2):
            next_fol_pos1 = np.array([fol_pos[2 * i][0] + fol_width * np.sin(angle_des),
                                        fol_pos[2 * i][1] - fol_width * np.cos(angle_des)])
            next_fol_pos2 = np.array([fol_pos[2 * i + 1][0] + fol_width * np.sin(angle_des),
                                        fol_pos[2 * i + 1][1] - fol_width * np.cos(angle_des)])
            fol_pos.append(next_fol_pos1)
            fol_pos.append(next_fol_pos2)

        return fol_pos     


    def _set_front_L_pos(self, world, F_pos, num_front_Ls):
        front_L_pos = []
        # 左側前方のフォロワを基準にする
        front_L_ref_coord = np.array([F_pos[-1][0], F_pos[-1][1]])
        L_width = world.followers[0].r_L["r5d"]
        # 台数分の初期位置を定義する
        for i in range(num_front_Ls):
            front_leader_next_coord = np.array([front_L_ref_coord[0] - i * L_width + np.random.rand()\
                                                    ,front_L_ref_coord[1] + L_width * 1.1]) 
            front_L_pos.append(front_leader_next_coord)
        
        return front_L_pos

    def _set_O_pos_st1(self, world, F_pos, des):
        # フォロワと目的地の中点
        # obstacle_ref_coord = F_pos[0] + 2 * (self.des - F_pos[0]) / 3
        # obstacle_ref_coord = F_pos[0] + (self.des - F_pos[0]) / 2
        O_ref_coord = F_pos[0] + 20
        O_pos = []
        for i in range(len(world.obstacles)):
            O_next_coord = np.array([O_ref_coord[0] + i * 3, O_ref_coord[1]]) 
            O_pos.append(O_next_coord)
        
        return O_pos
    
    def _set_O_pos_st2(self, world, F_pos, des):
        # フォロワと目的地の中点
        # obstacle_ref_coord = F_pos[0] + 2 * (self.des - F_pos[0]) / 3
        O_ref_coord = F_pos[0] + (des - F_pos[0]) / 2
        O_pos = []
        O_width = world.followers[0].r_F["r4"] * 6
        for i in range(len(world.obstacles)):
            O_next_coord = np.array([O_ref_coord[0] - np.random.rand() + i * O_width, 
                                     O_ref_coord[1] + (2 * np.random.rand() - 1)]) 
            O_pos.append(O_next_coord)
        
        return O_pos
    
    
    def _set_back_L_pos_st1(self, world, F_pos, num_back_Ls):
        back_L_pos = []
        # 一定の確率でリーダーをランダムに配置
        if np.random.rand() <= 0.5:
          # 右側後方のフォロワを基準にする
          back_L_ref_coord = np.array([F_pos[0][0], F_pos[0][1]])
          L_width = 1.75 * world.followers[0].r_L["r5d"]
          # 台数分の初期位置を定義する
          for i in range(num_back_Ls):
              back_L_next_coord = np.array([back_L_ref_coord[0] + i * L_width + np.random.rand()\
                                                      ,back_L_ref_coord[1] - L_width * 1.1]) 
              back_L_pos.append(back_L_next_coord)
        else: 
          # 左側前方のフォロワを基準にする
          back_L_ref_coord = np.array([F_pos[-1][0], F_pos[-1][1]])
          L_width = 1.75 * world.followers[0].r_L["r5d"]
          # 台数分の初期位置を定義する
          for i in range(num_back_Ls):
              back_L_next_coord = np.array([back_L_ref_coord[0] - i * L_width + np.random.rand()\
                                                      ,back_L_ref_coord[1] + L_width * 1.1]) 
              back_L_pos.append(back_L_next_coord)
        
        return back_L_pos

    def _set_back_L_pos_st2(self, world, F_pos, num_back_Ls):
        back_L_pos = []
        # 右側後方のフォロワを基準にする
        back_L_ref_coord = np.array([F_pos[0][0], F_pos[0][1]])
        L_width = 1.5 * world.followers[0].r_L["r5d"]
        # 台数分の初期位置を定義する
        for i in range(num_back_Ls):
            back_L_next_coord = np.array([back_L_ref_coord[0] + i * L_width + np.random.rand() 
                                        ,back_L_ref_coord[1] - L_width * 1.1])
            back_L_pos.append(back_L_next_coord)
        
        return back_L_pos


    def _set_circle_back_L_pos(self, world, ini_dis_to_des, angle_des):
        L_width = world.followers[0].r_L["r5d"] * 1.5
        F_width = world.followers[0].r_F["r4"]
        # 右側後方のフォロワを基準にする
        back_L_pos = [(ini_dis_to_des + F_width + L_width) * np.array([np.cos(angle_des), np.sin(angle_des)])]
        # 2台目
        back_L_next_coord = np.array([back_L_pos[0][0] + 2 * F_width * np.sin(angle_des)\
                                                ,back_L_pos[0][1] - 2 * F_width * np.cos(angle_des)]) 
        back_L_pos.append(back_L_next_coord)
        # 3台目
        back_L_next_coord = np.array([back_L_pos[0][0] - 2 * F_width * np.sin(angle_des)\
                                                ,back_L_pos[0][1] + 2 * F_width * np.cos(angle_des)]) 
        back_L_pos.append(back_L_next_coord)
        
        return back_L_pos
    
    def _rotate_axis(self, F_pos, F_width, angle):
        new_F_pos = [F_pos[0], np.array([F_pos[0][0] + F_width * np.sin(-angle),
                              F_pos[0][1] + F_width * np.cos(-angle)])]
        for i in range(2):
            next_fol_pos1 = np.array([new_F_pos[2 * i][0] + F_width * np.cos(angle),
                                        new_F_pos[2 * i][1] + F_width * np.sin(angle)])
            next_fol_pos2 = np.array([new_F_pos[2 * i + 1][0] + F_width * np.cos(angle),
                                        new_F_pos[2 * i + 1][1] + F_width * np.sin(angle)])
            new_F_pos.append(next_fol_pos1)
            new_F_pos.append(next_fol_pos2)
        
        return new_F_pos

    # return all agents that are back of swarm
    def _back_Ls(self, world):
        return [L for L in world.agents if not L.front]

    # return all front of swarm
    def _front_Ls(self, world):
        return [L for L in world.agents if L.front]

    ###########################################  For reward, observation, check_done ###########################################

    def _coord_trans(self, coord_bef: np.array) -> np.array:
        dis = LA.norm(coord_bef)
        angle = np.arctan2(coord_bef[1], coord_bef[0])
        angle /= np.pi  #  角度の正規化
        
        return np.array([dis, angle])
    
    def _calc_min_dis_to_F(self, agent, world):
        min_dis = np.inf
        # followerとの距離取得してminの距離を返す
        for F in world.followers:
            delta_pos = F.state.p_pos - agent.state.p_pos
            dist = LA.norm(delta_pos)
            if dist < min_dis: min_dis = dist
        
        return min_dis

    def _calc_Fs_dis_to_COM(self, world):
        F_COM = self.__calc_F_COM(world)
        Fs_dis_to_COM = []
        for F in world.followers:
            dis_to_COM = LA.norm(F_COM - F.state.p_pos)
            Fs_dis_to_COM.append(dis_to_COM)

        return Fs_dis_to_COM

    def _calc_Fs_dis_to_des(self, world, des):
        Fs_dis_to_des = []
        for F in world.followers:
            F_dis_to_des = LA.norm(des - F.state.p_pos)
            Fs_dis_to_des.append(F_dis_to_des)

        return Fs_dis_to_des

    def _calc_dis_to_des(self, world, des):
        F_COM = self._calc_F_COM(world)
        dis_to_des = LA.norm(F_COM - des)
        
        return dis_to_des


    def _calc_Fs_min_dis_to_O(self, world):
        min_dis_to_Os = []
        min_dis_to_O = np.inf
        for O in world.obstacles:
            for F in world.followers:
                F_dis_to_O = LA.norm(O.state.p_pos - F.state.p_pos)
                if F_dis_to_O < min_dis_to_O: min_dis_to_O = F_dis_to_O
        min_dis_to_Os.append(min_dis_to_O)
        
        return min_dis_to_Os

    def _calc_F_COM(self, world):
        F_sum = np.array([0.,0.])
        for F in world.followers:
            F_sum += F.state.p_pos
        F_COM = F_sum / len(world.followers)
        
        return F_COM

    def _check_col(self, L, world):
        is_col = False
        # リーダーとその他の物体との衝突のみ判定する
        # for obj_i in world.entities:
        #     for obj_j in world.entities:
        #         if obj_i is obj_j: continue
        #         delta_pos = obj_j.state.p_pos - obj_i.state.p_pos
        #         dist = LA.norm(delta_pos)
        #         dist_min = obj_i.size + obj_i.size
        #         if dist < dist_min:
        #             is_col = True
        #             break
        #     if is_col: break
        for obj_j in world.entities:
            if L is obj_j: continue
            dist = LA.norm(obj_j.state.p_pos - L.state.p_pos)
            dist_min = L.size + obj_j.size
            if dist < dist_min:
                is_col = True
                break
        
        return is_col

    def _check_goal(self, dis_to_des, rho_g):
        if (dis_to_des <= rho_g): is_goal = True
        else: is_goal = False
        
        return is_goal

    def _chech_div(self, world):
        Fs_pos = []
        for F in world.followers:
            Fs_pos.append(F.state.p_pos)
        
        dbscan = DBSCAN(eps = world.followers[0].r_F["r5"], min_samples = 2).fit(Fs_pos)
        y_dbscan = dbscan.labels_
        
        if len(np.where(y_dbscan ==-1)[0]) > 0 or np.max(y_dbscan) != 0: return True
        else: return False