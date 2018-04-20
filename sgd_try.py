from IRecom.GDSVD import *
from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np
from IRecom.functionalSVD import recommend
from MySQL.connection import getConnection


def __get_db__(update, min_d, new=14, __limit__=None):
    connection = getConnection()

    year_ago = datetime.now() - relativedelta(years=update)
    year_ago = year_ago.strftime("%Y-%m-%d")

    if new is None:
        if __limit__:
            query = "SELECT userProgramDownload_fkuser, userProgramDownload_fkprogram, userProgramDownload_count, " \
                    "userProgramDownload_updated " \
                    "FROM ivoox.userProgramDownloadLimited " \
                    "WHERE userProgramDownload_updated>'{}' " \
                    "AND  userProgramDownload_count>={} " \
                    "LIMIT {}"
            df = pd.read_sql(query.format(year_ago, min_d, __limit__), con=connection)
        else:
            query = "SELECT userProgramDownload_fkuser, userProgramDownload_fkprogram, userProgramDownload_count, " \
                    "userProgramDownload_updated " \
                    "FROM ivoox.userProgramDownloadLimited " \
                    "WHERE userProgramDownload_updated>'{}' " \
                    "AND  userProgramDownload_count>={}"
            df = pd.read_sql(query.format(year_ago, min_d, __limit__), con=connection)
    else:
        week_ago = datetime.now() - relativedelta(days=new)
        week_ago = week_ago.strftime("%Y-%m-%d")

        if __limit__:
            query = 'SELECT userProgramDownload_fkuser, userProgramDownload_fkprogram, userProgramDownload_count, ' \
                    'userProgramDownload_updated ' \
                    'FROM ivoox.userProgramDownloadLimited  ' \
                    'JOIN user ON user_id = userProgramDownload_fkuser ' \
                    'WHERE (userProgramDownload_updated> "{}"  ' \
                    'AND userProgramDownload_count>={}) ' \
                    'OR user_registerdate>= "{}" ' \
                    'LIMIT {}'

            df = pd.read_sql(query.format(year_ago, min_d, week_ago, __limit__), con=connection)

        else:
            query = 'SELECT userProgramDownload_fkuser, userProgramDownload_fkprogram, userProgramDownload_count, ' \
                    'userProgramDownload_updated ' \
                    'FROM ivoox.userProgramDownloadLimited ' \
                    'JOIN user ON user_id = userProgramDownload_fkuser ' \
                    'WHERE (userProgramDownload_updated> "{}" ' \
                    ' AND userProgramDownload_count>={}) ' \
                    'OR user_registerdate>= "{}"'

            df = pd.read_sql(query.format(year_ago, min_d, week_ago), con=connection)

    df.columns = ['user_id', 'program_id', 'download_count', 'download_updated']

    df["user_id"] = df["user_id"].astype(str)
    df["program_id"] = df["program_id"].astype(str)
    df['download_count'] = df['download_count'].astype(int)
    df['download_updated'] = pd.to_datetime(df['download_updated'], errors='coerce')
    return df


def __preprocess__(df, min_oyentes, date=None):
    if date is not None:
        programas_fecha = df[df['download_updated'] > date].program_id.unique()
        df = df[df.program_id.isin(programas_fecha)]

    count_prog = df.groupby('program_id')['user_id'].nunique()
    rango_oyentes = range(1, min_oyentes + 1)
    programas_oyentes = df[~df.program_id.isin(count_prog[count_prog.isin(rango_oyentes)].index)].program_id.unique()
    df = df[df.program_id.isin(programas_oyentes)]
    return df


def to_csv(r, name):
    """
    Function that saves the obtained recommender
    :param r: obtained dictionary.
    :param name: file name, where the recomendations are saved
    :return: None
    """
    header = '"row", "id_user", "Unique concatenate(RecommendedItem)"\n'
    with open(name, 'w') as f:
        f.write(header)
        k = 0
        for i in r.keys():
            f.write('"row{}", '.format(k) + i + ', "' + str(', '.join(r[i])) + '"' + '\n')
            k += 1


def run(__limit__=None):
    """
    Function for running the data
    :param __limit__: Limit in the number of rows asked in the MySQL query
    :return: None
    """
    t2 = time.time()
    min_d = 5
    years = 1
    days = 8
    last_listened_days = 40
    last_listened = datetime.now() - relativedelta(days=last_listened_days)
    if __limit__:
        df = __get_db__(years, min_d, new=days, __limit__=__limit__)
    else:
        df = __get_db__(years, min_d, new=days)
    df = __preprocess__(df, 20, last_listened)
    recomm_matrix = SGDSVD(0.01, 50, 20, 1)
    df['row'] = df.user_id.astype('category').cat.codes
    df['col'] = df.program_id.astype('category').cat.codes
    df['value'] = 1
    tr = Trainset(df)
    recomm_matrix.train(tr, v=1)
    item_u = np.array(sorted(df.program_id.unique()))
    user_u = np.array(sorted(df.user_id.unique()))
    user_batch = [user_u[75]]
    user_prog = {k: g['program_id'].tolist() for k, g in df[df.user_id == user_batch[0]].groupby('user_id')}
    print(recommend(recomm_matrix.p, recomm_matrix.q.T, user_batch, item_u, user_u, user_prog, 10))


if __name__ == '__main__':
    import time

    print(datetime.now())
    t1 = time.time()
    run(__limit__=1000000)
    # run()
    print('En total han hecho falta ' + str((time.time() - t1) // 60) + ' minutos y %.2f ' % ((time.time() - t1) % 60)
          + ' segundos.')
