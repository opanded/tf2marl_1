import os
import time
import pickle

import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import matplotlib.pyplot as plt

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
        # if not os.path.exists(os.path.join(save_dir)):
        #     os.makedirs(save_dir)
        # while os.path.exists(os.path.join(save_dir, exp_name)):
        #     print('WARNING: EXPERIMENT ALREADY EXISTS. APPENDING TO  TRIAL_NAME.')
        #     exp_name = exp_name + '_i'
        
        # 学習時と実行時でフォルダを分ける
        if not args["display"]:
            self.ex_path = os.path.join('learned_results', args["scenario_name"], str(_run._id))
            os.makedirs(self.ex_path, exist_ok=True)
            self.model_path = os.path.join(self.ex_path, 'models')
            os.makedirs(self.model_path, exist_ok=True)
            self.tb_path = os.path.join(self.ex_path, 'tb_logs')
            os.makedirs(self.tb_path, exist_ok=True)
            self.tb_writer = tf.summary.create_file_writer(self.tb_path)
            self.tb_writer.set_as_default()
        else:
            self.ex_path = os.path.join(args["restore_fp"].replace('/models', ''), "demo", str(_run._id))
            os.makedirs(self.ex_path, exist_ok=True)
            self.tb_path = os.path.join(self.ex_path, 'tb_logs')
            os.makedirs(self.tb_path, exist_ok=True)
            self.tb_writer = tf.summary.create_file_writer(self.tb_path)
            self.tb_writer.set_as_default()

        # save arguments
        args_file_name = os.path.join(self.ex_path, 'args.pkl')
        with open(args_file_name, 'wb') as fp:
            pickle.dump(args, fp)

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
        
        # for save mp4
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.all_frames = []

    def record_episode_end(self, agents):
        """
        Records an episode having ended.
        If save rate is reached, saves the models and prints some metrics.
        """
        self.episode_count += 1
        self.episode_step = 0
        self.episode_rewards.append(0.0)
        for ag_idx in range(self.n_agents):
            self.agent_rewards[ag_idx].append(0.0)

        if self.episode_count % (self.save_rate / 10) == 0:
            mean_rew = np.mean(self.episode_rewards[-self.save_rate // 10 : -1])
            self._run.log_scalar('traning.episode_reward', mean_rew, self.train_step)
            for ag_idx in range(self.n_agents):
                mean_ag_rew = np.mean(self.agent_rewards[ag_idx][:-self.save_rate//10:-1])
                self._run.log_scalar('traning.ep_rew_ag{}'.format(ag_idx), mean_ag_rew, self.train_step)



        if self.episode_count % self.save_rate == 0:
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
        print('...Finished total of {} episodes in {} minutes.'.format(self.episode_count,
                                                                       (time.time() - self.t_start) / 60))
        print(self._run._id)

    def convert(self, seconds):
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        
        return "%d:%02d:%02d" % (hour, minutes, seconds)
    
    def print_metrics(self):
        if self.n_adversaries == 0:
            print('steps: {}, episodes: {}, mean episode reward: {}, time: {}'.format(
                self.train_step, self.episode_count, round(np.mean(self.episode_rewards[-self.save_rate:-1]), 3),
                self.convert(round(time.time() - self.t_last_print, 3))))
        else:
            print('steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}'.format(
                self.train_step, self.episode_count, round(np.mean(self.episode_rewards[-self.save_rate:-1]), 3),
                [np.mean(rew[-self.save_rate:-1]) for rew in self.agent_rewards], self.convert(round(time.time() - self.t_last_print, 3))))
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
    
    def save_demo_result(self, pos_dict, reward_list_all):
        result_epi_dir = os.path.join(self.ex_path, "run_" + str(self.episode_count).zfill(2))
        os.makedirs(result_epi_dir, exist_ok = True)             
        # save mp4
        save = cv2.VideoWriter(str(result_epi_dir) + '/render.mp4', self.fourcc, 30.0, (850, 850))
        for img in self.all_frames:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            save.write(img)
        save.release()
        self.all_frames.clear()
        # save inital position                    
        result_pos_dir = os.path.join(result_epi_dir, "init_pos")
        os.makedirs(result_pos_dir, exist_ok = True)             
        for key, pos in pos_dict.items():
            with open(f'{str(result_pos_dir)}/{key}.csv', 'w') as pos_file:
                np.savetxt(pos_file, [len(pos)])
                np.savetxt(pos_file, pos)
        # save rewards
        result_rew_dir = os.path.join(result_epi_dir, "reward")
        os.makedirs(result_rew_dir, exist_ok = True)             
        header_list = ["R_F_far", "R_g", "R_div", "R_L_close", "R_back", "R_obs", "R_col"]
        reward_df = pd.DataFrame(reward_list_all)
        reward_df.to_csv(f"{str(result_rew_dir)}/reward_list.csv", index=False,\
                        header= header_list) 
        
        reward_list = []; reward_diff_list = []
        tmp_reward = 0
        for rew in reward_list_all:
            reward_sum = np.sum(rew)
            reward_diff = reward_sum - tmp_reward; tmp_reward = reward_sum
            reward_list.append(reward_sum)
            reward_diff_list.append(reward_diff)
        x = np.arange(0, len(reward_list), 1)
        # plot
        fig = plt.figure(figsize=(9.5, 10))
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.set_ylabel("Indivisual reward")
        # ax1.set_xlim(0, 50)
        ax1.grid()
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.set_ylabel("Whole reward")
        # ax1.set_xlim(0, 50)
        ax2.grid()
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.set_ylabel("Reward diff")
        # ax2.set_xlim(0, 50)
        ax3.grid()
        for i in range(len(header_list)):
            ax1.plot(x, reward_df[i], label = header_list[i], lw=1)
        ax2.plot(x, reward_list, label = "whole_reward", lw=2)
        ax3.plot(x, reward_diff_list, lw=2)
        fig.legend()
        fig.savefig(f"{result_epi_dir}/result.png")
        plt.get_current_fig_manager().window.wm_geometry("+1200+0")
        plt.show()


    @property
    def cur_episode_reward(self):
        return self.episode_rewards[-1]

    @cur_episode_reward.setter
    def cur_episode_reward(self, value):
        self.episode_rewards[-1] = value


