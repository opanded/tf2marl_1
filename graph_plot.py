#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

if __name__ == '__main__':
    # CSVファイルをUTF-8形式で読み込む
    # for i in range(3):
    #     data_des = pd.read_csv(f'./result/dis_to_des_{i}.csv',encoding = 'UTF8')
    #     data_agent = pd.read_csv(f'./result/dis_to_agent_{i}.csv',encoding = 'UTF8')
    #     des_y_data = []; agent_y_data = []

    #     fig = plt.figure(figsize=(12, 10))
    #     ax1 = fig.add_subplot(1, 2, 1)
    #     ax1.set_ylabel("distance to goal")
    #     ax1.set_xlim(0, 50)
    #     # ax1.set_ylim(0, 1.5)
    #     ax2 = fig.add_subplot(1, 2, 2)
    #     ax2.set_ylabel("distance to closest agent")
    #     ax2.set_xlim(0, 50)
    #     # ax2.set_ylim(0, 0.5)
        
    #     x = np.arange(0, 50, 1)
    #     for j in range(3):
    #         des_y_data.append(list(data_des[f"{j}"]))
    #         agent_y_data.append(list(data_agent[f"{j}"]))
            
    #         ax1.plot(x, des_y_data[j], label=f"agent{j}")
    #         ax2.plot(x, agent_y_data[j])
    #     # 横線
    #     ax1.hlines(0.4, -100, 100, colors='blue', linestyle='dashed'); ax1.hlines(0.25, -100, 100, colors='red', linestyle='dashed'); ax1.hlines(0.15, -100, 100, colors='green', linestyle='dashed'); ax1.hlines(0.55, -100, 100, colors='green', linestyle='dashed')
    #     ax2.hlines(0.4, -100, 100, colors='blue', linestyle='dashed'); ax2.hlines(0.25, -100, 100, colors='red', linestyle='dashed') ; ax2.hlines(0.15, -100, 100, colors='green', linestyle='dashed'); ax2.hlines(0.55, -100, 100, colors='green', linestyle='dashed')
    #     fig.legend()
    #     plt.show()

    # reward_df = pd.read_csv(f'./learned_policy/sheperding_multi_leaders/2_leaders/try3/result/0004/reward_list.csv',
    #                         encoding = 'UTF8')
    # header_list = ["R_F_far", "R_g", "R_div", "R_L_close", "R_obs", "R_col"]
                        
    # reward_list = []; reward_diff_list = []
    # tmp_reward = 0
    # for index, row in reward_df.iterrows():
    #     reward_sum = np.sum(row)
    #     reward_diff = reward_sum - tmp_reward; tmp_reward = reward_sum
    #     reward_list.append(reward_sum)
    #     reward_diff_list.append(reward_diff)
    # x = np.arange(0, len(reward_list), 1)
    # # plot
    # fig = plt.figure(figsize=(9.5, 10))
    # ax1 = fig.add_subplot(3, 1, 1)
    # ax1.set_ylabel("Indivisual reward")
    # # ax1.set_xlim(0, 50)
    # ax1.grid()
    # ax2 = fig.add_subplot(3, 1, 2)
    # ax2.set_ylabel("Whole reward")
    # # ax1.set_xlim(0, 50)
    # ax2.grid()
    # ax3 = fig.add_subplot(3, 1, 3)
    # ax3.set_ylabel("Reward diff")
    # # ax2.set_xlim(0, 50)
    # ax3.grid()
    # for i in range(len(header_list)):
    #     # print(reward_df[header_list[i]])
    #     ax1.plot(x, reward_df[header_list[i]], label = header_list[i], lw=1.5)
    # ax2.plot(x, reward_list, label = "whole_reward", lw=2)
    # ax3.plot(x, reward_diff_list, lw=2)
    # fig.legend()
    # fig.savefig(f"./result.png")
    # plt.show()

    with open('learned_results/sheperding_st2/maddpg_LSTM16/move_3Os/1/rewards.pkl', 'rb') as reward_file:
        result = pickle.load(reward_file)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111) 
        x = np.arange(0, len(result) * 1000, 1000)
        ax.plot(x, result, label= "mean reward")
        ax.set_ylim(-10, 100)
        ax.set_xlabel("episode", fontsize=24); ax.set_ylabel("mean reward", fontsize=24)
        ax.legend()
        plt.tick_params(labelsize=18)
        plt.show()