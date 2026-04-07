import os.path
import pandas as pd
import numpy as np
import random


def makeMissingData(path, missing_ratio, seed, missingness='MCAR'):
    df = pd.read_csv(path)
    return makeMissingDataByDf(df=df, missing_ratio=missing_ratio, seed=seed, missingness=missingness)



def makeMissingDataByDf(df, missing_ratio, seed, missingness='MCAR'):
    np.random.seed(seed)

    print("seed and rate:", seed, missing_ratio)
    
    mask = np.random.rand(*df.shape)
    df2 = df.copy()
    
    # if missingness=='MAR':
    #     v=1
    #     ranked_df = df.rank(axis=0, ascending=False)
    #     tmp = np.empty(df.shape)
    #     for i in range(df.shape[1]):
    #         tmp[:,i]=ranked_df.iloc[:,v].values / ranked_df.iloc[:,v].sum()
        
    #     mask = mask + tmp*32
    #     threshold = np.percentile(mask, missing_ratio*100)
    #     mask[mask < threshold] = 0
    #     mask[mask != 0] = 1

    # elif missingness=='MNAR':
    #     ranked_df = df.rank(axis=0, ascending=False)
    #     tmp = np.empty(df.shape)
    #     for i in range(df.shape[1]):
    #         tmp[:,i]=ranked_df.iloc[:,i].values / ranked_df.iloc[:,i].sum()

    #     mask = mask + tmp*32
    #     threshold = np.percentile(mask, missing_ratio*100)
    #     mask[mask < threshold] = 0
    #     mask[mask != 0] = 1
    
    # else:
    mask[mask <= missing_ratio] = 0     # masked
    mask[mask != 0] = 1                 # remained
    
    df2[mask == 0] = np.nan
    print(df.shape[0]*df.shape[1], (1-mask).sum())

    return df, df2





