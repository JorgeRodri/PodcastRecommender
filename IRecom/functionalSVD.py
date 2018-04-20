import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import numpy as np
import time
from Testing.MSE import mse


def get_table(df, __ones__=False):
    """
    transforms data into a pivot table, users times programs
    :param df: data
    :param __ones__: if True all values will be set to 1, else values = download_count
    :return: df with rows and cols f each element, pivot table
    """
    user_u = list(sorted(df.user_id.unique()))
    item_u = list(sorted(df.program_id.unique()))

    row = df.user_id.astype('category').cat.codes
    col = df.program_id.astype('category').cat.codes

    df['row'] = row
    df['col'] = col

    if not __ones__:
        data = df['download_count'].tolist()
        table = csr_matrix((data, (row, col)), shape=(len(user_u), len(item_u)))
    else:
        data = np.ones(df['download_count'].shape)
        table = csr_matrix((data, (row, col)), shape=(len(user_u), len(item_u)))

    return df, table


def normalize(table, _technique='row'):
    """
    normalization function for the users times programs listened table
    :param table: table to be normalized
    :param _technique: technique to be used: row, weighted, max, others return the table
    :return: normalized table
    """
    if _technique == 'row':
        ccd = sparse.spdiags(1. / table.sum(1).T, 0, *[table.shape[0], table.shape[0]])
        return ccd * table
    elif _technique == 'weighted':
        nc = table.sum()
        wm = table.sum(axis=1) / nc
        wn = table.sum(axis=0) / nc
        return table / nc - np.dot(wm, wn)
    elif _technique == 'max':
        maxs = 1 / table.max(1).todense()
        ccd = sparse.spdiags(np.asarray(maxs).reshape(-1), 0, *[table.shape[0], table.shape[0]])
        return ccd * table
    else:
        return table


def __get_dict__(df):
    d = {k: g['program_id'].tolist() for k, g in df.groupby('user_id')}
    return d


def recommend_whole(df, U, sigma, V, k, num_batches=10):
    """
    Function that finds the recomendation from the user times programs times downloads table
    :param df: dataframe with data
    :param U: U from the SVD
    :param sigma: sigma from SVD
    :param V: V from SVD
    :param k: number of recomendations for each users
    :param num_batches: number of batches in which the recomender should be trained, recommended >10
    :return: dictionary with the recomendations for each user
    """
    St = np.dot(sigma, V) #rapido

    item_u = np.array(sorted(df.program_id.unique()))
    user_u = np.array(sorted(df.user_id.unique()))

    if num_batches == 1:
        batches = user_u
    else:
        batches = np.array_split(user_u, num_batches)
    all_recom = dict()

    user_prog = __get_dict__(df) #es rapido, para lo que es

    for i in range(num_batches):
        user_batch = batches[i]
        # t1 = time.time()
        recom = recommend(U, St, user_batch, item_u, user_u, user_prog, k) #creo que se puede mejorar
        # t2 = time.time()
        # print('Time for a Batch: %.2f' % (t2-t1))
        # meter quitar los escuchados
        all_recom.update(recom)
    return all_recom


def recommend(U, St, user_batch, item_u, user_u, user_prog, k):
    """
    Recomend to batch of users the top k programs
    :param U: U from SVD
    :param St: Sigma*V from SVD
    :param user_batch: users for which recommend programs
    :param item_u: programs in data
    :param user_u: all users in data
    :param user_prog: programs that users have listened
    :param k: number of recommendations
    :return: small dictionary with recomendations for user in batch
    """
    #     predictions, user, programs, original, row, item_u, k=5
    # Get and sort the user's predictions
    user_row_number = np.nonzero(np.isin(user_u, user_batch))[0]
    predicted = np.dot(U[user_row_number[0]:user_row_number[-1]+1, :], St)
    prog = dict()

    for i in range(predicted.shape[0]): #es necesario el for?
        listened_prog = user_prog[user_batch[i]]
        u_cols = np.nonzero(np.isin(item_u, listened_prog))
        current = predicted[i, :]
        current[u_cols] = -1
        ind = current.argsort()[-1:-k - 1:-1]
        prog[user_batch[i]] = item_u[list(ind)]
    return prog


def predict_rating(listened, user, U, SV):
    return U[user, :].dot(SV[:, listened])


def evaluate_batch(batch, U, SV, lrate):
    for user in range(batch.shape[0]):
        programs_in_user = np.nonzero(batch[user, :])[1]
        for listened in programs_in_user:
            err = lrate * (batch[user, listened] - predict_rating(listened, user, U, SV))

            U[user, :] += err * SV[:, listened]
            SV[:, listened] += err * U[user, :]
    return U, SV


def svd_gd(df, k, num_epochs, lrate, initial_values_U=None, initial_values_SV=None, l=None, __ones__=True):
    user_u = list(sorted(df.user_id.unique()))
    item_u = list(sorted(df.program_id.unique()))

    df, table = get_table(df, __ones__=__ones__)
    if not __ones__:
        table = normalize(table, _technique='max')

    if initial_values_U is not None:
        U = initial_values_U
    else:
        U = np.ones((len(user_u), k), dtype=float) * 0.1
    if initial_values_SV is not None:
        SV = initial_values_SV
    else:
        SV = np.ones((k, len(item_u)), dtype=float) * 0.1

    if l is None:
        l = []

    rows, cols = table.nonzero()

    for epoch in range(num_epochs):
        print('Actual epoch :' + str(epoch))
        time1 = time.time()

        for user, listened in zip(rows, cols):
            err = lrate * (table[user, listened] - predict_rating(listened, user, U, SV))

            U[user, :] += err * SV[:, listened]
            SV[:, listened] += err * U[user, :]

        t = time.time() - time1
        print('Enlapsed time for the last epoch: %.2f seconds.' % t)
        error = mse(table, U, SV)
        print('Mean Square Error of: %.4f' % error)
        l.append(error)
    return U, SV, table, l

