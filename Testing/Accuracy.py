import pandas as pd
from tqdm import tqdm
import numpy as np
from math import floor


def __get_testing_programs__(df, m=1):
    users = df.user_id.unique().tolist()
    hidden = pd.DataFrame()
    for u in tqdm(users):
        a = df[df['user_id'] == u].sample(m)
        df = df.drop(a.index)
        hidden = hidden.append(a)
    return df, hidden


def __get_testing_one__(df):
    users = df.user_id.unique().tolist()
    h = pd.DataFrame()
    for u in tqdm(users):
        m = sum(df['user_id'] == u)
        a = df[df['user_id'] == u].sample(m - 1)
        h = h.append(a)
        df = df.drop(a.index)
    return df, h


def __get_testing_high_prog__(df, m):
    users = df.user_id.unique().tolist()
    hidden = pd.DataFrame()
    for u in tqdm(users):
        d = df[df['user_id'] == u]
        a = df[df['user_id'] == u].sort_values('download_count', ascending=False).iloc[:m, :]
        df = df.drop(a.index)
        hidden = hidden.append(a)
    return df, hidden


def __get_testing_high_one__(df):
    users = df.user_id.unique().tolist()
    h = pd.DataFrame()
    for u in tqdm(users):
        d = df[df['user_id'] == u]
        m = sum(df['user_id'] == u)
        a = df[df['user_id'] == u].sort_values('download_count', ascending=False).iloc[:m - 1, :]
        h = h.append(a)
        df = df.drop(a.index)
    return df, h


def get_sample(df, p, ivoox=None):
    """
    Get a sample of users from data equal to a number or a proportion
    :param df: Data to extract user sample from
    :param p: Proportion of users if 0<p<1
    :param ivoox: if ivoox used, includes the users in ivoox
    :return: sample of users
    """
    users = df['user_id'].unique()
    sample = np.random.choice(users, floor(len(users) * p), replace=False)
    if ivoox:
        sample = np.append(sample, ivoox)
    return sample


def get_test(df, p, m=2, random=True):
    """
    Divide data into two, test and train
    :param df: data to be divided
    :param p: porportion of testing
    :param m: number of programs to eliminate
    :param random: If the selection of erased programs is random if False the most listened programs would be eliminated
    :return: train data, test data
    """
    #     not_sampled = df[~df['user_id'].isin(sample)].user_id.unique()
    users = df['user_id'].unique()
    user_test = np.random.choice(users, floor(len(users) * p), replace=False)
    df1 = df[df['user_id'].isin(user_test)]
    df = df[~df['user_id'].isin(user_test)]

    one = df1[df1['user_id'].isin(
        df1['user_id'].value_counts(ascending=True).keys()[df1['user_id'].value_counts(ascending=True) <= m])]
    two = df1[df1['user_id'].isin(
        df1['user_id'].value_counts(ascending=True).keys()[df1['user_id'].value_counts(ascending=True) > m])]

    print('Obteniendo partición de test y el ground true, puede tomar algún tiempo.')
    if random:
        one, test1 = __get_testing_one__(one)
        two, test2 = __get_testing_programs__(two, m)
    else:
        one, test1 = __get_testing_high_one__(one)
        two, test2 = __get_testing_high_prog__(two, m)
    df = df.append(one)
    df = df.append(two)
    return df, test1.append(test2)

