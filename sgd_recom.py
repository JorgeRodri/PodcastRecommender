from IRecom.GDSVD import *
from datetime import datetime
import pandas as pd
import numpy as np
from IRecom.functionalSVD import recommend
from MySQL.connection import getConnection
import json


def query_to_df(credenciales, query):
    connection = getConnection(credenciales)
    df = pd.read_sql(query, connection)
    connection.close()
    return df


def __get_db__(date_updated,  min_d, __limit__=None):
    credentials_dir = 'C:\\Users\\jorge.rodriguez\\ivoox\\ivooxrepo\\python_ai\\MySQL\\credenciales_corrector.json'
    with open(credentials_dir, 'r') as f:
        connection_credentials = json.loads(f.read())
    connection = getConnection(connection_credentials)

    query = 'SELECT userProgramDownload_fkuser, userProgramDownload_fkprogram, userProgramDownload_count, ' \
            'userProgramDownload_updated ' \
            'FROM ivoox.userProgramDownloadLimited  ' \
            'JOIN user ON user_id = userProgramDownload_fkuser ' \
            'WHERE (userProgramDownload_updated> "{}"  ' \
            'AND userProgramDownload_count>={}) ' \
            'LIMIT {}'

    df = pd.read_sql(query.format(date_updated, min_d, __limit__), con=connection)

    df.columns = ['user_id', 'program_id', 'download_count', 'download_updated']

    df["user_id"] = df["user_id"].astype(str)
    df["program_id"] = df["program_id"].astype(str)
    df['download_count'] = df['download_count'].astype(int)
    df['download_updated'] = pd.to_datetime(df['download_updated'], errors='coerce')
    return df


def __process_data__(df, min_oyentes, date=None):
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

    columns = 'programs_id, programs_name, programs_description, programs_fkcategory'
    credentials_dir = 'C:\\Users\\jorge.rodriguez\\ivoox\\ivooxrepo\\python_ai\\MySQL\\credenciales_corrector.json'
    with open(credentials_dir, 'r') as f:
        arch = json.loads(f.read())
    programs = query_to_df(arch, 'SELECT %s FROM ivoox.programs' % columns)
    program_name = {row['programs_id']: row['programs_name'] for index, row in programs.iterrows()}

    min_d = 1  # for WSGD use 1, uses number of downloads as a kind of rating
    update = datetime(2018, 4, 1)
    if __limit__:
        df = __get_db__(update, min_d, __limit__=__limit__)
    else:
        df = __get_db__(update, min_d)
    df = __process_data__(df, 20, date=None)
    recomm_matrix = WSGDSVD(0.01, 50, 20, 1, 10, 5)
    df['row'] = df.user_id.astype('category').cat.codes
    df['col'] = df.program_id.astype('category').cat.codes
    df['value'] = 1  # df.download_count
    tr = Trainset(df)
    recomm_matrix.train(tr, v=1)
    item_u = np.array(sorted(df.program_id.unique()))
    user_u = np.array(sorted(df.user_id.unique()))
    user_batch = [user_u[55]]
    user_prog = {k: g['program_id'].tolist() for k, g in df[df.user_id == user_batch[0]].groupby('user_id')}
    print([program_name[int(i)] for i in list(user_prog.values())[0]])
    print([program_name[int(i)] for i in list(
        recommend(recomm_matrix.p, recomm_matrix.q.T, user_batch, item_u, user_u, user_prog, 10)
                                                                                                .values())[0]])
    print('Número de programas: %d. Número de usuarios: %d.' % (item_u.shape[0], user_u.shape[0]))


if __name__ == '__main__':
    import time

    print(datetime.now())
    t1 = time.time()
    run(__limit__=100000)
    # run()
    print('En total han hecho falta ' + str((time.time() - t1) // 60) + ' minutos y %.2f ' % ((time.time() - t1) % 60)
          + ' segundos.')
