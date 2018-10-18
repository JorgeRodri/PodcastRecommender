import numpy as np
import time
from IRecom.functionalSVD import __get_dict__, recommend


class SGDSVD:
    """A basic rating prediction algorithm based on matrix factorization."""

    def __init__(self, learning_rate, n_epochs, n_factors, reg_coef):

        self.lr = learning_rate  # learning rate for SGD
        self.n_epochs = n_epochs  # number of iterations of SGD
        self.n_factors = n_factors  # number of factors
        self.reg = reg_coef
        self.p = None
        self.q = None
        self.w = None
        self.w_i = None
        self.w_u = None
        self.train_set = None

    def train(self, train_set, v=0):
        """Learn the vectors p_u and q_i with SGD"""

        print('Fitting data with SGD...')

        # Randomly initialize the user and item factors.
        if self.p and self.q and self.w and self.w_i and self.w_u:
            p = self.p
            q = self.q
            w = self.w
            w_i = self.w_i
            w_u = self.w_u
        else:
            p = np.random.normal(0, .1, (train_set.n_users, self.n_factors))
            q = np.random.normal(0, .1, (train_set.n_items, self.n_factors))
            w = train_set.df.value.mean()
            w_i = np.random.normal(0, .1, train_set.n_items)
            w_u = np.random.normal(0, .1, train_set.n_users)
        k = 0

        # SGD procedure
        for _ in range(self.n_epochs):
            # TODO: Incorporate Alpha and Epsilon for implicit information.
            k += 1
            t1 = time.time()
            for u, i, r_ui in train_set.all_ratings():
                mu = np.abs(np.random.normal(0, 1))
                err = r_ui - (np.dot(p[u], q[i]) + w + w_i[i] + w_u[u])
                # Update vectors p_u and q_i
                p[u] += self.lr * mu * err * q[i] - self.reg * p[u]
                q[i] += self.lr * mu * err * p[u] - self.reg * q[i]
                w_i[i] += self.lr * mu * err - self.reg * w_i[i]
                w_u[u] += self.lr * mu * err - self.reg * w_u[u]
                # Note: in the update of q_i, we should actually use the previous (non-updated) value of p_u.
                # In practice it makes almost no difference.
            if v == 1:
                t = time.time() - t1
                print('Epoch {} complete.'.format(k))
                print('Time needed, {} seconds.'.format(t))
            elif v == 2:
                pass

        self.p, self.q = p, q
        self.w, self.w_i, self.w_u = w, w_i, w_u
        self.train_set = train_set

    def estimate(self, u, i):
        """Return the estimated rating of user u for item i."""

        # return scalar product between p_u and q_i if user and item are known,
        # else return the average of all ratings
        if self.train_set.knows_user(u) and self.train_set.knows_item(i):
            return np.dot(self.p[u], self.q[i]) + self.w + self.w_i + self.w_u
        else:
            return self.train_set.global_mean


class Trainset:
    def __init__(self, df):
        self.df = df
        self.n_users = df.user_id.nunique()
        self.n_items = df.program_id.nunique()

    def all_ratings(self):
        subset = self.df[['row', 'col', 'value']]
        return list(zip(*[self.df[c].values.tolist() for c in subset]))

    def knows_user(self, u):
        return u in self.df.user_id.values

    def knows_item(self, i):
        return i in self.df.program_id


class WSGDSVD:
    """A basic rating prediction algorithm based on matrix factorization."""

    def __init__(self, learning_rate, n_epochs, n_factors, reg_coef, alpha, epsilon):

        self.lr = learning_rate  # learning rate for SGD
        self.n_epochs = n_epochs  # number of iterations of SGD
        self.n_factors = n_factors  # number of factors
        self.reg = reg_coef
        self.alpha = alpha
        self.epsilon = epsilon
        self.p = None
        self.q = None
        self.w = None
        self.w_i = None
        self.w_u = None
        self.train_set = None

    def train(self, train_set, v=0):
        """Learn the vectors p_u and q_i with SGD"""

        print('Fitting data with SGD...')

        # Randomly initialize the user and item factors.
        if self.p and self.q and self.w and self.w_i and self.w_u:
            p = self.p
            q = self.q
            w = self.w
            w_i = self.w_i
            w_u = self.w_u
        else:
            p = np.random.normal(0, .1, (train_set.n_users, self.n_factors))
            q = np.random.normal(0, .1, (train_set.n_items, self.n_factors))
            w = train_set.df.value.mean()
            w_i = np.random.normal(0, .1, train_set.n_items)
            w_u = np.random.normal(0, .1, train_set.n_users)
        k = 0

        # SGD procedure
        for _ in range(self.n_epochs):
            k += 1
            t1 = time.time()
            for u, i, r_ui in train_set.all_ratings():
                c_ui = 1 + self.alpha*np.log(1+self.epsilon**(-1)*r_ui)
                l_ui = 1 if r_ui > 0 else 0
                mu = np.abs(np.random.normal(0, 1))
                err = c_ui*(l_ui - (np.dot(p[u], q[i]) + w + w_i[i] + w_u[u]))
                # Update vectors p_u and q_i
                p[u] += self.lr * mu * err * q[i] - self.reg * p[u]
                q[i] += self.lr * mu * err * p[u] - self.reg * q[i]
                w_i[i] += self.lr * mu * err - self.reg * w_i[i]
                w_u[u] += self.lr * mu * err - self.reg * w_u[u]
                # Note: in the update of q_i, we should actually use the previous (non-updated) value of p_u.
                # In practice it makes almost no difference.
            if v == 1:
                t = time.time() - t1
                print('Epoch {} complete.'.format(k))
                print('Time needed, {} seconds.'.format(t))
            elif v == 2:
                pass

        self.p, self.q = p, q
        self.w, self.w_i, self.w_u = w, w_i, w_u
        self.train_set = train_set

    def estimate(self, u, i):
        """Return the estimated rating of user u for item i."""

        # return scalar product between p_u and q_i if user and item are known,
        # else return the average of all ratings
        if self.train_set.knows_user(u) and self.train_set.knows_item(i):
            return np.dot(self.p[u], self.q[i]) + self.w + self.w_i[i] + self.w_u[u]
        else:
            return self.train_set.global_mean


class ALS:
    """A basic rating prediction algorithm based on matrix factorization."""

    def __init__(self, n_epochs, n_factors, reg_coef):

        self.n_epochs = n_epochs  # number of iterations of SGD
        self.n_factors = n_factors  # number of factors
        self.reg = reg_coef
        self.p = None
        self.q = None
        self.train_set = None

    def train(self, train_set, v=0):
        """Learn the vectors p_u and q_i with SGD"""

        print('Fitting data with SGD...')

        # Randomly initialize the user and item factors.
        if self.p and self.q:
            p = self.p
            q = self.q
        else:
            p = np.random.normal(0, .1, (train_set.n_users, self.n_factors))
            q = np.random.normal(0, .1, (train_set.n_items, self.n_factors))
        k = 0

        # ALS procedure
        for _ in range(self.n_epochs):
            # TODO: Incorporate Alpha and Epsilon for implicit information, i.e. WALS
            k += 1
            t1 = time.time()
            # Precompute Y-transpose-Y and X-transpose-X
            xtx = p.T.dot(p)
            yty = q.T.dot(q)
            y = np.linalg.solve(yty + self.reg * np.eye(self.n_factors),
                                np.dot(train_set.table.T, p))
            x = np.linalg.solve(xtx + self.reg * np.eye(self.n_factors),
                                np.dot(train_set.table, q))

            if v == 1:
                t = time.time() - t1
                print('Epoch {} complete.'.format(k))
                print('Time needed, {} seconds.'.format(t))
            elif v == 2:
                pass

            p, q = x, y
        self.p, self.q = p, q


def recommend_whole(df, U, St, k, num_batches=10):
    """
    Function that finds the recomendation from the user times programs times downloads table
    :param df: dataframe with data
    :param U: U from the SVD
    :param St: sigma*V in the SVD decomposition
    :param k: number of recomendations for each users
    :param num_batches: number of batches in which the recomender should be trained, recommended >10
    :return: dictionary with the recomendations for each user
    """

    item_u = np.array(sorted(df.program_id.unique()))
    user_u = np.array(sorted(df.user_id.unique()))

    if num_batches == 1:
        batches = user_u
    else:
        batches = np.array_split(user_u, num_batches)
    all_recom = dict()

    user_prog = __get_dict__(df)  # es rapido, para lo que es

    for i in range(num_batches):
        user_batch = batches[i]
        # t1 = time.time()
        recom = recommend(U, St, user_batch, item_u, user_u, user_prog, k)  # creo que se puede mejorar
        # t2 = time.time()
        # print('Time for a Batch: %.2f' % (t2-t1))
        # meter quitar los escuchados
        all_recom.update(recom)
    return all_recom
