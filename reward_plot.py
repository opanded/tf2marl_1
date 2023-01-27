#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # CSVファイルをUTF-8形式で読み込む
    data_st1 = pd.read_csv(f'learned_results/stage2/any_Fs/4h_4-9Fs_no_R_back/stage2.csv'
                           ,encoding = 'UTF8')
    data_st2 = pd.read_csv(f'learned_results/stage3/any_Fs/10h_4-9Fs_final_no_R_back/stage3.csv'
                           ,encoding = 'UTF8')
    data_from_scratch = pd.read_csv(f'learned_results/stage3/any_Fs/14h_final_no_R_back/from_scratch.csv'
                                    ,encoding = 'UTF8')

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 15)
    # # ax1.set_ylim(0, 1.5)
    ax.plot(list(data_st1.iloc[:, 1]), list(data_st1.iloc[:, 2])
            , label="stage1")
    ax.plot(list(data_st2.iloc[:, 1]), list(data_st2.iloc[:, 2])
            , label="stage2")
    # ax.plot(list(data_from_scratch.iloc[:, 1]), list(data_from_scratch.iloc[:, 5])
    #         , label="from_scratch")
    ax.set_xlabel("Time", fontsize=24); ax.set_ylabel("Mean reward", fontsize=24)
    ax.grid()
    ax.tick_params(labelsize=18)

    fig.legend(fontsize=18)
    fig.savefig(f"./reward_result.png")
    plt.show()