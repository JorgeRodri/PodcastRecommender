import json
import pymysql
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.sparse as sparse


arch = '../MySQL/credenciales_corrector.json'


def getconnection(server_data):
    """
    parameters already fixed inside the function
    :return: the connection to the server
    """
    connection = pymysql.connect(charset='utf8', cursorclass=pymysql.cursors.DictCursor, **server_data)
    return connection


def query_to_df(credenciales, query):
    with open(credenciales, 'r') as f:
        connection_credentials = json.loads(f.read())
    connection = getconnection(connection_credentials)
    df = pd.read_sql(query, connection)
    connection.close()
    return df


def open_process(dir_list):
    dfdl = pd.DataFrame()
    for i in tqdm(dir_list):
        temp_df = pd.read_pickle(i)
        if len(temp_df) > 0:
            temp_df = temp_df[temp_df['countryCode'] == 'ES']
            dfdl = dfdl.append(temp_df[['userId', 'programId']])
    return dfdl


if __name__ == '__main__':
    data_path = 'C:\\Users\\jorge.rodriguez\\Documents\\Jorge\\data'

    columns = 'programs_id, programs_name, programs_description, programs_fkcategory'
    programs = query_to_df(arch, 'SELECT %s FROM ivoox.programs' % columns)
    program_name = {row['programs_id']: row['programs_name'] for index, row in programs.iterrows()}

    # list_of_dir = list(map(lambda x: os.path.join(data_path, x), list(filter(lambda x: 'mongo_downloads2018' == x[:19],
    #                                                                          os.listdir(data_path)))))
    # df = open_process(list_of_dir)
    df = pd.read_pickle('dataset.pkl')
    # print('Memory taken by the DataFrame: ' + str(sys.getsizeof(df) // (2 ** 20)) + 'MB')
    # df['value'] = 1
    # final = df[df['userId'] != 0].groupby(['userId', 'programId']).agg('count')
    #
    # df = pd.DataFrame()
    #
    # df['user'], df['program'] = map(list, zip(*np.array(final.index)))
    # df['value'] = np.array(final.value)

    user_u = np.array(sorted(df.user.unique()))
    item_u = np.array(sorted(df.program.unique()))

    # df['row'] = df.user.astype('category').cat.codes
    # df['col'] = df.program.astype('category').cat.codes

    table = sparse.csr_matrix((df['value'], (df['row'], df['col'])), shape=(len(user_u), len(item_u)))

    import implicit

    alpha = 15
    user_vecs, item_vecs = implicit.alternating_least_squares((table * alpha).astype('double'),
                                                              # show_progress=False,
                                                              factors=20,
                                                              regularization=20,
                                                              iterations=50)
    # Todo: use the AlternatingLeastSquares class instead, good results

    print(user_vecs.shape)
    print(item_vecs.shape)

    user = 6776060  # user_u[666]

    user_cols = df[df.user == user].col.values
    print(list(map(lambda x: program_name[x], item_u[user_cols])))

    u = np.where(user_u == user)[0][0]

    r_hat = np.dot(user_vecs[u], item_vecs.T)
    print(list(map(lambda x: program_name[x], item_u[r_hat.argsort()[-1:-8:-1]])))