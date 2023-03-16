import os
import time
import pickle

import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RLLogger(object):
    def __init__(self, exp_name, _run, n_agents, n_adversaries, save_rate):
        '''
        Initializes a logger.
        This logger will take care of results, and debug info, but never the replay buffer.
        '''
        self._run = _run
        args = _run.config
        self.n_agents = n_agents
        self.n_adversaries = n_adversaries
        
        # 学習時と実行時でフォルダを分ける
        if (not args["display"]) and (not args["evaluate"]):
            self.ex_path = os.path.join(args["save_path"], str(_run._id))
            os.makedirs(self.ex_path, exist_ok=True)
            self.model_path = os.path.join(self.ex_path, 'models')
            os.makedirs(self.model_path, exist_ok=True)
            self.tb_path = os.path.join(self.ex_path, 'tb_logs')
            os.makedirs(self.tb_path, exist_ok=True)
            self.tb_writer = tf.summary.create_file_writer(self.tb_path)
            self.tb_writer.set_as_default()
        else:
            if args["evaluate"]:
                self.ex_path = os.path.join(args["restore_fp"].replace('/models', ''), "eval", str(_run._id))
            else:
                self.ex_path = os.path.join(args["restore_fp"].replace('/models', ''), "demo", str(_run._id))
            os.makedirs(self.ex_path, exist_ok=True)
            # self.tb_path = os.path.join(self.ex_path, 'tb_logs')
            # os.makedirs(self.tb_path, exist_ok=True)
            # self.tb_writer = tf.summary.create_file_writer(self.tb_path)
            # self.tb_writer.set_as_default()

        # save arguments
        args_file_name = os.path.join(self.ex_path, 'args.pkl')
        with open(args_file_name, 'wb') as fp:
            pickle.dump(args, fp)

        self.num_episodes = args["num_episodes"]
        self.episode_rewards = [0.0]
        self.agent_rewards = [[0.0] for _ in range(n_agents)]
        self.final_ep_rewards = []  # sum of rewards for training curve
        self.final_ep_ag_rewards = []  # agent rewards for training curve
        self.agent_info = [[[]]]  # placeholder for benchmarking info
        self.train_step = 0
        self.episode_step = 0
        self.episode_count = 0
        self.t_start = time.time()
        self.t_last_print = time.time()

        self.save_rate = save_rate
        self.save_rate_time = []
        self.learning_time = args["learning_time_log"]
        
        # for save mp4
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.all_frames = []
        
        # for evaluation
        self.num_success = 0
        self.num_divide = 0
        self.num_exceed = 0
        self.num_collide = 0
        self.num_over = 0

    def record_episode_end(self, agents, display, evaluate):
        """
        Records an episode having ended.
        If save rate is reached, saves the models and prints some metrics.
        """
        self.episode_count += 1
        self.episode_step = 0
        self.episode_rewards.append(0.0)
        for ag_idx in range(self.n_agents):
            self.agent_rewards[ag_idx].append(0.0)

        if (not display) and (not evaluate):
            # episodeではなく10分おきに保存するようにする．
            # save_rateの10倍の頻度
            # if self.episode_count % (self.save_rate / 10) == 0:
            if (time.time() - self.t_last_print) >= 10 * 60:
                mean_rew = np.mean(self.episode_rewards[-self.save_rate // 10 : -1])
                self._run.log_scalar('traning.episode_reward', mean_rew)
                for ag_idx in range(self.n_agents):
                    mean_ag_rew = np.mean(self.agent_rewards[ag_idx][:-self.save_rate//10:-1])
                    self._run.log_scalar('traning.ep_rew_ag{}'.format(ag_idx), mean_ag_rew)
            # if self.episode_count % self.save_rate == 0:
                self.print_metrics()
                self.calculate_means()
                self.save_models(agents)

    def experiment_end(self):
        rew_file_name = os.path.join(self.ex_path, 'rewards.pkl')
        with open(rew_file_name, 'wb') as fp:
            pickle.dump(self.final_ep_rewards, fp)
        agrew_file_name = os.path.join(self.ex_path, 'agrewards.pkl')
        with open(agrew_file_name, 'wb') as fp:
            pickle.dump(self.final_ep_ag_rewards, fp)
        
        # rewardをplotする．
        result = self.final_ep_rewards
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        # 時間のx軸を作成する
        x = np.linspace(0, self.learning_time, len(result))
        ax.plot(x, result)
        # ax.set_ylim(-10, 100)
        ax.set_xlabel("Time", fontsize=24); ax.set_ylabel("Mean reward", fontsize=24)
        ax.grid()
        ax.legend()
        plt.tick_params(labelsize=18)
        fig.savefig(f"{self.ex_path}/reward_result.png")
        
        print('...Finished total of {} episodes in {}.'.format(self.episode_count,
                                                                self.convert(time.time() - self.t_start)))
        print(self._run._id)

    def convert(self, seconds):
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        
        return "%d:%02d:%02d" % (hour, minutes, seconds)
    
    def print_metrics(self):
        mean_episode_reward = round(np.mean(self.episode_rewards[-self.save_rate:-1]), 3)
        taken_time = time.time() - self.t_last_print
        total_time= time.time() - self.t_start
        self.save_rate_time.append(taken_time)
        ave_time = np.mean(self.save_rate_time)
        time_left = self.convert((self.num_episodes - self.episode_count) / self.save_rate * ave_time)
        if self.n_adversaries == 0:
            print('episodes: {}, mean reward: {}, time: {}, total time: {}, time left: {}'.format(
                self.episode_count, mean_episode_reward, self.convert(taken_time), self.convert(total_time) , time_left))
        else:
            print('steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}'.format(
                self.train_step, self.episode_count, round(np.mean(self.episode_rewards[-self.save_rate:-1]), 3),
                [np.mean(rew[-self.save_rate:-1]) for rew in self.agent_rewards], 
                self.convert(taken_time)))
        self.t_last_print = time.time()

    def save_models(self, agents):
        for idx, agent in enumerate(agents):
            agent.save(os.path.join(self.model_path, 'agent_{}'.format(idx)))

    def calculate_means(self):
        self.final_ep_rewards.append(np.mean(self.episode_rewards[-self.save_rate:-1]))
        for ag_rew in self.agent_rewards:
            self.final_ep_ag_rewards.append(np.mean(ag_rew[-self.save_rate:-1]))


    def add_agent_info(self, agent, info):
        raise NotImplementedError()

    def get_sacred_results(self):
        return np.array(self.episode_rewards), np.array(self.agent_rewards)
    
    def draw_pos_fig(self, pos_list, num_Ls, num_Fs, num_Os, dest, rho_g, Os_info, save_dir):
        data_pos = np.array(pos_list)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(1, 1, 1)
        mid_idx = int(len(data_pos) / 2)
        ax.set_xlim(-8.5, 8.5)
        ax.set_ylim(-5, 12.5)
        goal = patches.Circle(xy=dest, radius=rho_g, fc='gray')
        ax.add_patch(goal)
        # leader
        for i in range(num_Ls):
            idx = i * 2
            radius = 0.075
            col = "r"
            if i == 0:
                ax.plot(list(data_pos[0:, idx]), list(data_pos[0:, idx + 1])
                        , linewidth=0.75, color="r", label="Leader")
            else:
                ax.plot(list(data_pos[0:, idx]), list(data_pos[0:, idx + 1])
                        , linewidth=0.75, color="r")
            L_ini = patches.Circle(xy=(data_pos[0, idx], data_pos[0, idx + 1]), radius=radius, fc=col)
            L_fi = patches.Circle(xy=(data_pos[-1, idx], data_pos[-1, idx + 1]), radius=radius, fc=col)
            L_mid = patches.Circle(xy=(data_pos[mid_idx, idx], data_pos[mid_idx, idx + 1]), radius=radius, fc=col)
            ax.add_patch(L_ini); ax.add_patch(L_mid); ax.add_patch(L_fi)
        # follower
        for j in range(num_Fs):
            idx = (num_Ls * 2) + j * 2
            radius = 0.075
            col = "b"
            if j == 0:
                ax.plot(list(data_pos[0:, idx]), list(data_pos[0:, idx + 1])
                    , linewidth=0.75, color="b", label="Follower")
            else: 
                ax.plot(list(data_pos[0:, idx]), list(data_pos[0:, idx + 1])
                    , linewidth=0.75, color="b")
            F_ini = patches.Circle(xy=(data_pos[0, idx], data_pos[0, idx + 1]), radius=radius, fc=col)
            F_fi = patches.Circle(xy=(data_pos[-1, idx], data_pos[-1, idx + 1]), radius=radius, fc=col)
            F_mid = patches.Circle(xy=(data_pos[mid_idx, idx], data_pos[mid_idx, idx + 1]), radius=radius, fc=col)
            ax.add_patch(F_ini); ax.add_patch(F_mid); ax.add_patch(F_fi)
        # COM
        idx = (num_Ls + num_Fs) * 2
        col = "g"
        F_COM_ini = patches.Rectangle(xy=(data_pos[0, idx], data_pos[0, idx + 1]), width=0.15, height=0.15, fc=col)
        F_COM_fi = patches.Rectangle(xy=(data_pos[-1, idx], data_pos[-1, idx + 1]), width=0.15, height=0.15, fc=col)
        F_COM_mid = patches.Rectangle(xy=(data_pos[mid_idx, idx], data_pos[mid_idx, idx + 1]), width=0.15, height=0.15, fc=col)
        ax.add_patch(F_COM_ini); ax.add_patch(F_COM_mid); ax.add_patch(F_COM_fi)
        # obstacle
        for k, O in enumerate(Os_info):
            # remain環境について
            if O.have_vel:
                O_range = patches.Circle(xy=O.init_pos, radius=O.max_range, fc="lightgreen")
                ax.add_patch(O_range)
            # trajectoryのplot
            idx = ((num_Ls + num_Fs + 1) * 2) + k * 2 
            radius = O.size
            col = "g"
            if k == 0:
                ax.plot(list(data_pos[0:, idx]), list(data_pos[0:, idx + 1])
                    , linewidth=0.75, color="g", label="Obstacle")
            else:
                ax.plot(list(data_pos[0:, idx]), list(data_pos[0:, idx + 1])
                    , linewidth=0.75, color="g") 
            O_ini = patches.Circle(xy=(data_pos[0, idx], data_pos[0, idx + 1]), radius=radius, fc=col)
            O_fi = patches.Circle(xy=(data_pos[-1, idx], data_pos[-1, idx + 1]), radius=radius, fc=col)
            O_mid = patches.Circle(xy=(data_pos[mid_idx, idx], data_pos[mid_idx, idx + 1]), radius=radius, fc=col)
            ax.add_patch(O_ini); ax.add_patch(O_mid); ax.add_patch(O_fi)
        ax.set_xlabel("x", fontsize=24); ax.set_ylabel("y", fontsize=24)
        ax.set_aspect('equal')
        ax.grid()
        fig.legend(fontsize=18)
        fig.savefig(f"{save_dir}/positions.png") 
    
    def save_demo_result(self, pos_list, obj_info, dest_info, Os_info, reward_list_all):
        result_epi_dir = os.path.join(self.ex_path, "run_" + str(self.episode_count).zfill(2))
        os.makedirs(result_epi_dir, exist_ok = True)
        result_fig_dir = os.path.join(result_epi_dir, "snapshots")
        os.makedirs(result_fig_dir, exist_ok = True)    
        # save snapshot and movie
        idx1 = int(len(self.all_frames) / 3); idx2 = idx1 * 2 
        save = cv2.VideoWriter(str(result_epi_dir) + '/render.mp4', self.fourcc, 30.0, (600, 600))
        for i, img in enumerate(self.all_frames):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if (i == 0) or (i==idx1) or (i==idx2) or (i==len(self.all_frames)-1):
                save_path = str(result_fig_dir) + f'/idx{i}.png'
                cv2.imwrite(save_path, img)
            save.write(img)
        save.release()
        self.all_frames.clear()
        # draw position fig
        self.draw_pos_fig(pos_list, *obj_info, *dest_info, Os_info, result_epi_dir)
        # save rewards
        result_rew_dir = os.path.join(result_epi_dir, "reward")
        os.makedirs(result_rew_dir, exist_ok = True)             
        header_list = ["R_F_far", "R_g", "R_div", "R_L_close", "R_back", "R_obs", "R_col"]
        reward_df = []
        for i in range(self.n_agents):
            reward_df.append(pd.DataFrame(reward_list_all[i]))
            reward_df[i].to_csv(f"{str(result_rew_dir)}/agent{i}_reward.csv", index=False,\
                            header= header_list) 
        
        reward_list = [[] for _ in range(self.n_agents)]
        # tmp_reward = 0
        fig = plt.figure(figsize=(9.5, 10))
        ax_list = [fig.add_subplot(4, 1, 1 * (idx+1)) for idx in range(2 * self.n_agents)]
        for idx in range(self.n_agents):
            for rew in reward_list_all[idx]:
                reward_sum = np.sum(rew)
                # reward_diff = reward_sum - tmp_reward; tmp_reward = reward_sum
                reward_list[idx].append(reward_sum)
                # reward_diff_list.append(reward_diff)
            x = np.arange(0, len(reward_list[idx]), 1)
            # plot
            ax_list[2 * idx].set_ylabel(f"agent{idx}_reward")
            # ax_list1.set_xlim(0, 50)
            ax_list[2 * idx].grid()
            ax_list[2 * idx + 1].set_ylabel(f"agent{idx}_whole_reward")
            # ax_list1.set_xlim(0, 50)
            ax_list[2 * idx + 1].grid()
            if idx == 0:
                for i in range(len(header_list)):
                    ax_list[2 * idx].plot(x, reward_df[idx][i], label = header_list[i], lw=1)
                ax_list[2 * idx + 1].plot(x, reward_list[idx], label = "whole_reward", lw=2)
            else:
                for i in range(len(header_list)):
                    ax_list[2 * idx].plot(x, reward_df[idx][i], lw=1)
                ax_list[2 * idx + 1].plot(x, reward_list[idx], lw=2)    
        fig.legend()
        fig.savefig(f"{result_epi_dir}/result.png")
        plt.get_current_fig_manager().window.wm_geometry("+1200+0")
        plt.show()
        
    def save_eval_result(self, info_n, num_eval_episodes):          
        if "goal" in info_n:
            self.num_success += 1
        elif "divide" in info_n:
            self._run.log_scalar('done_info', f"{self.episode_count}: divide")
            self.num_divide += 1
        elif "collide" in info_n:
            self._run.log_scalar('done_info', f"{self.episode_count}: collide")
            self.num_collide += 1
        else:
            self._run.log_scalar('done_info', f"{self.episode_count}: over")
            self.num_over += 1
        if self.episode_count + 1 >= num_eval_episodes:
            success_rate = 100 * self.num_success / (self.episode_count + 1)
            divide_rate = 100 * self.num_divide / (self.episode_count + 1)
            collide_rate = 100 * self.num_collide / (self.episode_count + 1)
            over_rate = 100 * self.num_over / (self.episode_count + 1)
            self._run.log_scalar('done_info', 
                                 f"success_rate: {success_rate}% collide_rate: {collide_rate}% divide_rate: {divide_rate}%  over_rate: {over_rate}%")
            print(f"success_rate: {success_rate}% collide_rate: {collide_rate}% divide_rate: {divide_rate}% over_rate: {over_rate}%")
    @property
    def cur_episode_reward(self):
        return self.episode_rewards[-1]

    @cur_episode_reward.setter
    def cur_episode_reward(self, value):
        self.episode_rewards[-1] = value


