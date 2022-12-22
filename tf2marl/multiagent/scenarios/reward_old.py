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
    
    
    
    
    ############ observationのoldを保存
    L_to_Fs = []
        cross_list = []
        L_to_COM = self.F_COM - L.state.p_pos
        for F in world.followers:
            L_to_F = F.state.p_pos - L.state.p_pos
            L_to_F_unit = (F.state.p_pos - L.state.p_pos) / LA.norm(L_to_F)
            cross_list.append(np.cross(L_to_COM, L_to_F_unit))
            L_to_F = self.funcs._coord_trans(L_to_F)
            L_to_F[0] /= self.max_dis_to_agent  if L_to_F[0] < self.max_dis_to_agent else L_to_F[0]  # 正規化
            L_to_Fs.append(L_to_F)
        L_to_COM = self.funcs._coord_trans(L_to_COM)
        L_to_COM[0] /= self.max_dis_to_agent  if L_to_COM[0] < self.max_dis_to_agent else L_to_COM[0]  # 正規化
        # to do: フォロワの並び替えの方法もう少し考える．
        # self.L_to_Fs_deques[i].append([L_to_COM, L_to_Fs[np.argmin(cross_list)], L_to_Fs[np.argmax(cross_list)]])
        L_to_Fs = np.array(L_to_Fs)
        L_to_Fs = L_to_Fs[np.argsort(L_to_Fs[:, 0])]
        self.L_to_Fs_deques[i].append(L_to_Fs.tolist())