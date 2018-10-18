import json
import pymysql
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.utils.extmath import safe_sparse_dot

arch = 'C:/Users/jorge.rodriguez/PycharmProjects/iVooxPython/MySQL/credenciales_corrector.json'


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


def ji_cosine_similarity(X, Y=None, dense_output=True):
    X, Y = check_pairwise_arrays(X, Y)

    X_normalized = normalize(X, norm='l1', copy=True)

    K = safe_sparse_dot(X_normalized, Y.T,
                        dense_output=dense_output)
    return K


class JIRecommender:
    def __init__(self, df):

        self.df = df
        self.table = None
        self.items = None
        self.user = None
        self.processed_df = None
        self.similarity = None

    def get_table(self, category=None):

        if category is not None:
            sus_cat = self.df[self.df['cat'] == 11].copy()
        else:
            sus_cat = self.df.copy()
        user_u = list(sorted(sus_cat['suscription_fkuser'].unique()))
        item_u = list(sorted(sus_cat['suscription_filtervalue'].unique()))

        row = sus_cat['suscription_fkuser'].astype('category').cat.codes
        col = sus_cat['suscription_filtervalue'].astype('category').cat.codes

        sus_cat['row'] = row
        sus_cat['col'] = col

        data = sus_cat['value'].tolist()
        self.processed_df = sus_cat
        self.table = csr_matrix((data, (row, col)), shape=(len(user_u), len(item_u)))
        self.items = item_u
        self.user = user_u

    def fit(self, dense=False):
        self.similarity = ji_cosine_similarity(self.table.T, dense_output=dense)
        return self.similarity

    def retrieve_top(self, program, program_name, k=10):
        if type(program) == str:
            indx = self.processed_df[
                        self.processed_df.suscription_filtervalue.map(program_name) == program]['col'].iloc[0]
        else:
            indx = self.processed_df[self.processed_df.suscription_filtervalue == program]['col'].iloc[0]
        top = list(map(lambda x: program_name[x] if x != 0 else '',
                       np.array(self.items)[
                           np.reshape(np.array(np.argsort(self.similarity[indx].todense())[0, -1:-k - 1:-1]), k)]))
        values = self.similarity[indx].todense()[
                            0, np.reshape(np.array(np.argsort(self.similarity[indx].todense())[0, -1:-k - 1:-1]), k)]
        return top, values


if __name__ == '__main__':
    columns = 'programs_id, programs_name, programs_description, programs_fkcategory'
    programs = query_to_df(arch, 'SELECT %s FROM ivoox.programs' % columns)
    program_name = {row['programs_id']: row['programs_name'] for index, row in programs.iterrows()}
    program_cat = {row['programs_id']: row['programs_fkcategory'] for index, row in programs.iterrows()}

    columns = 'suscription_fkuser, suscription_filtervalue, suscription_filtertext, suscription_sendnotification'
    suscriptions = query_to_df(arch, 'SELECT %s FROM ivoox.suscription where suscription_date> "2018-07-01"' % columns)

    suscriptions['value'] = 1
    suscriptions['cat'] = suscriptions['suscription_filtervalue'].map(program_cat)

    recommender = JIRecommender(suscriptions)
    recommender.get_table(11)
    recommender.fit()
    a, b = recommender.retrieve_top(163595, program_name, 10)
    print(a)
    print(b)

