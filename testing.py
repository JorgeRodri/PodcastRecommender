from IRecom.functionalSVD import *
from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np
from scipy.sparse import linalg
import time
from MySQL.connection import getConnection
from tqdm import tqdm
from Testing.Accuracy import get_sample, get_test


def test_recommender(r, truth):
    n = 0
    k = 0
    o = 0
    for u in tqdm(list(truth.user_id.unique())):
        truth_u = truth[truth.user_id == u]
        truth_u = truth_u.program_id.tolist()

        predictions = r[u]
        if len(truth_u) == 1:
            o += 1
            if len(set(predictions).intersection(truth_u))>=1:
                n += 1
                k += 1
        elif len(set(predictions).intersection(truth_u))>=1:
            n += 1
    return n/len(truth.user_id.unique())*100, k/o*100, (n-k)/(len(truth.user_id.unique())-o)*100


def test_k_norm(df, truth, K, N):
    accs = []
    accs_ones = []
    not_ones_accs = []
    for k in K:
        for norm in N:
            print('\n------------------------------------------------------------\n')
            print('Calculating for k='+str(k)+' and norm: '+norm)
            t1 = time.time()
            if norm == 'No':
                df, table = get_table(df, __ones__=True)
            else:
                df, table = get_table(df, __ones__=False)
                table = normalize(table, _technique=norm)
            U, s, V = linalg.svds(table, k=k)
            sigma = np.diag(s)
            t2 = time.time()
            r = recommend_whole(df, U, sigma, V, k=10, num_batches=10)
            print('Time to get the recommendation: '+str((time.time()-t2)//60) + ' minutes and ' +
                  str((time.time()-t2) % 60)+' seconds, getting the test.')
            acc, acc_one, acc_no_one = test_recommender(r, truth)
            accs.append(acc)
            accs_ones.append(acc_one)
            not_ones_accs.append(acc_no_one)
            print('Process finished, time needed for this iteration: ' + str((time.time()-t1)//60) + ' minutes and ' +
                  str((time.time()-t1) % 60)+' seconds, getting the test results.')
            print('Total Accuracy of: '+str(acc))
            print('One listened Accuracy of: '+str(acc_one))
            print('Accuracy for more than one litened: '+str(acc_no_one))

    return accs, accs_ones, not_ones_accs


connection = getConnection()

query = "SELECT userProgramDownload_fkuser, userProgramDownload_fkprogram, userProgramDownload_count, userProgramDownload_updated FROM ivoox.userProgramDownloadLimited WHERE userProgramDownload_updated> '2017-01-01' AND  userProgramDownload_count>4"

print('Iniciando, cargamos datos')
# df = pd.read_csv('data/user_program_count.csv', header=0)
df = pd.read_sql(query, con=connection)
df.columns = ['user_id', 'program_id', 'download_count', 'download_updated']

print('Elegimos los tipos de datos')
df["user_id"] = df["user_id"].astype(str)
df["program_id"] = df["program_id"].astype(str)
df['download_count'] = df['download_count'].astype(int)
df['download_updated'] = pd.to_datetime(df['download_updated'], errors='coerce')

print('Limpiamos la base de datos')
year_ago = datetime.now() - relativedelta(years=1)
programas_fecha = df[df['download_updated'] > year_ago].program_id.unique()

count_prog = df.groupby('program_id')['user_id'].nunique()

df = df[df.program_id.isin(programas_fecha)]
rango_oyentes = range(1, 11)
programas_oyentes = df[~df.program_id.isin(count_prog[count_prog.isin(rango_oyentes)].index)].program_id.unique()
df = df[df.program_id.isin(programas_oyentes)]


df = df[df['download_updated'] > year_ago]

count_prog = df.groupby('program_id')['user_id'].nunique()
programas_oyentes = df[~df.program_id.isin(count_prog[count_prog.isin(rango_oyentes)].index)].program_id.unique()
df = df[df.program_id.isin(programas_oyentes)]


df = df[df['user_id'].isin(
        df['user_id'].value_counts(ascending=True).keys()[df['user_id'].value_counts(ascending=True) > 1])]
sample = get_sample(df, 0.1)
df = df[df.user_id.isin(sample)]

print('Obtenemos el test set')
df, truth = get_test(df, 0.1, m=2, random=True)

if __name__ == '__main__':
    N = ['max']
    K = [5, 20, 50, 100, 200]
    print('\nCalculamos la eficacia de los parametros\n')
    acc, acc_one, acc_no_one = test_k_norm(df, truth, K, N)
