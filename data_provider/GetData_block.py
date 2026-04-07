import os.path
import pandas as pd
import numpy as np
import random


def makeMissingData(path, missing_ratio, seed, missingness='MCAR'):
    df = pd.read_csv(path)
    return makeMissingDataByDf(df=df, missing_ratio=missing_ratio, seed=seed, missingness=missingness)



def makeMissingDataByDf(df, missing_ratio, seed, missingness='MCAR'):
    np.random.seed(seed)
    print("seed and rate:", seed, missing_ratio, missingness)

    df2 = df.copy()
    n_total = df.size  # 总元素个数
    n_missing_target = int(missing_ratio * n_total)  # 目标缺失数

    current_missing = 0
    n_rows = df.shape[0]

    while current_missing < n_missing_target:
        # 随机选择一列
        col = np.random.choice(df.columns)

        # 随机选择一个起始行，使得可以构成长度为10的区间
        if n_rows < 10:
            raise ValueError("DataFrame行数少于10，无法形成长度为10的区间。")

        start_idx = np.random.randint(0, n_rows - 10 + 1)
        end_idx = start_idx + 10

        # 计算这一段中“尚未为NaN”的元素数量
        not_na_mask = df2.loc[start_idx:end_idx - 1, col].notna()
        newly_missing = not_na_mask.sum()

        # 设置为缺失
        df2.loc[start_idx:end_idx - 1, col] = np.nan

        # 更新已缺失的总数
        current_missing += newly_missing

    print(f"Missing inserted: {current_missing}/{n_total} ({current_missing / n_total:.2%})")

    return df, df2





