from IRecom.functionalSVD import get_table, recommend_whole, normalize
from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np
from scipy.sparse import linalg
from MySQL.connection import getConnection
import pymysql


def __get_local_db__(file):
    if file is None:
        return pd.DataFrame()
    df = pd.read_csv(file)
    df = df[['userProgramDownload_fkuser',  'userProgramDownload_fkprogram',
             'userProgramDownload_count', 'userProgramDownload_updated']]
    df.columns = ['user_id', 'program_id', 'download_count', 'download_updated']

    df["user_id"] = df["user_id"].astype(str)
    df["program_id"] = df["program_id"].astype(str)
    df['download_count'] = df['download_count'].astype(int)
    df['download_updated'] = pd.to_datetime(df['download_updated'], errors='coerce')
    return df


def __get_db__(update, min_d, new=14):
    connection = getConnection()

    year_ago = datetime.now() - relativedelta(years=update)
    year_ago = year_ago.strftime("%Y-%m-%d")

    week_ago = datetime.now() - relativedelta(days=new)
    week_ago = week_ago.strftime("%Y-%m-%d")

    query = 'SELECT userProgramDownload_fkuser, userProgramDownload_fkprogram, userProgramDownload_count, ' \
                    'userProgramDownload_updated ' \
                    'FROM ivoox.userProgramDownloadLimited  ' \
                    'JOIN user ON user_id = userProgramDownload_fkuser ' \
                    'WHERE (userProgramDownload_updated> "{}"  ' \
                    'AND userProgramDownload_count>={}) ' \
                    'OR user_registerdate>= "{}" '

    df = pd.read_sql(query.format(year_ago, min_d, week_ago), con=connection)

    df.columns = ['user_id', 'program_id', 'download_count', 'download_updated']

    df["user_id"] = df["user_id"].astype(str)
    df["program_id"] = df["program_id"].astype(str)
    df['download_count'] = df['download_count'].astype(int)
    df['download_updated'] = pd.to_datetime(df['download_updated'], errors='coerce')
    return df


def get_db(file=None, **kwargs):
    try:
        df = __get_db__(**kwargs)
    except pymysql.DataError as e:
        print("DataError")
        print(e)
        df = __get_local_db__(file)
    except pymysql.InternalError as e:
        print("InternalError")
        print(e)
        df = __get_local_db__(file)
    except pymysql.IntegrityError as e:
        print("IntegrityError")
        print(e)
        df = __get_local_db__(file)
    except pymysql.OperationalError as e:
        print("OperationalError")
        print(e)
        df = __get_local_db__(file)
    except pymysql.NotSupportedError as e:
        print("NotSupportedError")
        print(e)
        df = __get_local_db__(file)
    except pymysql.ProgrammingError as e:
        print("ProgrammingError")
        print(e)
        df = __get_local_db__(file)
    except Exception as e:
        print(e)
        print("Unknown error occurred")
        df = __get_local_db__(file)
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
    header = '"row ID", "id_user", "Unique concatenate(RecommendedItem)"\n'
    with open(name, 'w') as f:
        f.write(header)
        k = 0
        for i in r.keys():
            f.write('"row{}", '.format(k) + i + ', "' + str(', '.join(r[i])) + '"' + '\n')
            k += 1


def run():
    """
    Function for running the recommender
    """
    min_d = 5
    years = 1
    days = 8
    last_listened_days = 40
    last_listened = datetime.now() - relativedelta(days=last_listened_days)
    last_listened = datetime(2018, 1, 1, 0, 0, 1)
    k = 200
    num_recom = 10

    print('Getting data')

    file = 'data/downloads_user_program_20180111.csv'
    df = get_db(file=file, update=years, min_d=min_d, new=days)
    print('Preprocessing')
    df = __preprocess__(df, 20, last_listened)
    df, table = get_table(df, __ones__=True)
    table = normalize(table, _technique='No')
    U, s, V = linalg.svds(table, k=k)
    sigma = np.diag(s)
    print('Obtaining recommendations')
    r = recommend_whole(df, U, sigma, V, k=num_recom, num_batches=5000)

    to_csv(r, 'recommender_jorge_{}_{}.csv'.format(k, num_recom))


if __name__ == '__main__':
    run()
