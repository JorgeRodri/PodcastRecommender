import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, linalg
import numpy as np


class Recomendador:
    algorithm = 'SVD'

    def __init__(self, data_dir=None, programas_dir=None):
        '''
        Constructor del objeto recommendador
        data_dir: directorio con el csv para los datos
        k: número componentes para el SVD
        programas_dir: directorio para programas
        '''

        if data_dir:
            self.original = pd.read_csv(data_dir, header=0)
            self.original.columns = ['user_id', 'program_id', 'download_count', 'download_updated']
        if programas_dir:
            self.programas_df = pd.read_csv(programas_dir, header=0, error_bad_lines=False, warn_bad_lines=False)
            self.programas_df.columns = ['id', 'name', 'recomendations', 'category', 'subcat', 'audios', 'last_update']
            self.programas_df = self.programas_df[['id', 'name', 'category', 'subcat']]
        self.description = "Sistema de recomendación basado en SVD para ivoox"
        self.author = "Jorge Rodriguez Molinuevo"

    def tipos_correctos(self):
        '''Define cada variable dandole el tipo correcto de valor en la tabla de usuarios programas escuchas '''

        self.original["user_id"] = self.original["user_id"].astype(str)
        self.original["program_id"] = self.original["program_id"].astype(str)
        self.original['download_count'] = self.original['download_count'].astype(int)
        self.original['download_updated'] = pd.to_datetime(self.original['download_updated'], errors='coerce')
        self.users = self.original['user_id'].unique()
        self.programs = self.original['program_id'].unique()

    def tipos_programas(self):
        '''Define cada variable dandole el tipo correcto de valor en la tabla de programas '''

        self.programas_df['id'] = self.programas_df['id'].astype(str)
        self.programas_df['name'] = self.programas_df['name'].astype(str)
        self.programas_df['category'] = self.programas_df['category'].astype('category')
        self.programas_df['subcat'] = self.programas_df['subcat'].astype('category')
        self.programas_df = self.programas_df[self.programas_df.id.isin(self.original.program_id.unique())]

    def preprocess(self, date=datetime.now() - relativedelta(years=2), k=10):
        '''
        Transformación de los datos de para reducir las dimensiones de los datos y mejorar la preción del recomendador
        eliminando ruido en la matriz, como programas con pocos oyentes.
        '''
        # Primero eliminamos programas de sobra
        entradas_origen = len(self.original)
        count_prog = self.original.groupby('program_id')['user_id'].nunique()
        programas_fecha = self.original[self.original[
                                            'download_updated'] > date].program_id.unique()  # programas que se han escuchado una vez desde 2015
        self.original = self.original[self.original.program_id.isin(programas_fecha)]
        rango_oyentes = range(1, k + 1)
        programas_oyentes = self.original[~self.original.program_id.isin(count_prog[count_prog.isin(
            rango_oyentes)].index)].program_id.unique()  # programas que solo son escuchados entre 1 y 10
        self.original = self.original[self.original.program_id.isin(programas_oyentes)]

        # Ahora se eliminan usuarios
        usuarios_fecha = self.original[self.original[
                                           'download_updated'] > date].user_id.unique()  # programas que se han escuchado una vez desde 2015
        self.original = self.original[self.original.user_id.isin(usuarios_fecha)]

        # eliminamos de nuevo programas con poca audiencia en caso de que los usuarios eliminados hallan podido cambiar la distribución
        rango_oyentes = range(1, 11)
        count_prog = self.original.groupby('program_id')['user_id'].nunique()
        programas_oyentes = self.original[
            ~self.original.program_id.isin(count_prog[count_prog.isin(rango_oyentes)].index)].program_id.unique()

        print('Antes teníamos ' + str(len(self.users)) + ', ahora tenemos un total de ' + str(
            len(self.original.user_id.unique())) + ' usuarios.')
        print('Antes teníamos ' + str(len(self.programs)) + ', ahora tenemos un total de ' + str(
            len(self.original.program_id.unique())) + ' programas.')
        print('Ésto implica que de una base de datos de ' + str(
            entradas_origen) + ' de entradas hemos pasado a una con ' + str(len(self.original)) + ' entradas.')
        self.users = self.original['user_id'].unique()
        self.programs = self.original['program_id'].unique()

    def __get_table__(self, __ones__=False):
        '''
        Función para obtener la tabla de usuarios por programas
        df: es el dataframe con los datos
        __ones__: variable que define si usar 1s en vez de las descargas
        '''

        user_u = list(sorted(self.original.user_id.unique()))
        item_u = list(sorted(self.original.program_id.unique()))

        row = self.original.user_id.astype('category').cat.codes
        col = self.original.program_id.astype('category').cat.codes

        self.original['row'] = row
        self.original['col'] = col

        if not __ones__:
            data = self.original['download_count'].tolist()
        else:
            data = np.ones(self.original['download_count'].shape)
        table = csr_matrix((data, (row, col)), shape=(len(user_u), len(item_u)))

        self.row = row
        self.col = col
        self.table = table
        return row, col, table

    def normalize(self, _technique='row'):
        '''
        Función utilizada en el momento de querer normalizar los datos por ahora se han ofrecido 2 técnicas
        pero ninguna ofrece mejore resultados que el no utilizar nada, se mantiene el método a la espera de
        encontrar alguna técnica que mejore los resultados
        _technique define la técnica
        '''

        if _technique == 'row':
            ccd = sparse.spdiags(1. / self.table.sum(1).T, 0, *[table.shape[0], table.shape[0]])
            self.table = ccd * self.table
        elif _technique == 'weighted':
            nc = self.table.sum()
            wm = self.table.sum(axis=1) / nc
            wn = self.table.sum(axis=0) / nc
            self.table = self.table / nc - np.dot(wm, wn)
        else:
            return 0

    def svd(self, k=None):
        '''
        Descomposición en valores singulares utilizando scipy.linalg.svds para poder reducir la matriz de dimensión
        k: número de componentes principales
        '''
        if not k:
            self.k = 5
        else:
            self.k = k

        row, col, table = self.__get_table__

        U, s, V = linalg.svds(self.table, k=self.k)
        sigma = np.diag(s)
        self.U = U
        self.sigma = sigma
        self.V = V
        return U, sigma, V

    def __recommend_batch__(self, user_batch, rows, cols, user_prog, item_u, k):
        '''
        Obtener las recomendacones de un batch, función privada
        '''
        #     predictions, user, programs, original, row, item_u, k=5
        # Get and sort the user's predictions
        aux = rows.user_id.isin(user_batch)
        user_row_number = rows[
            aux].row.values  # índice en la matriz del usuario, ver si coinviden con los usuarios(justo abajo)
        users = rows[aux].user_id.values
        predicted = np.dot(self.U[user_row_number, :], self.St)
        prog = dict()

        for i in range(len(predicted)):
            listened_prog = user_prog[user_prog.user_id == users[i]].program_id.values
            u_cols = cols[cols.program_id.isin(listened_prog)].col
            current = predicted[i, :]
            current[u_cols] = 0
            ind = current.argsort()[-1:-k - 1:-1].invers()
            prog[users[i]] = item_u[list(ind)]
        return prog

    def recommend_whole(self, k=5, num_batches=1):
        '''
        Halla el diccionario con las recomendaciones para todos los usuarios.
        :param k: numero de recomendaciones
        :param num_batches: numero de batches en los que se separan los usuarios
        :return: diccionario: {Usuario: k recomendaciones}
        '''

        self.St = np.dot(self.sigma, self.V)

        rows = self.original[['user_id', 'row']].drop_duplicates()
        cols = self.original[['program_id', 'col']].drop_duplicates()

        item_u = np.array(sorted(self.original.program_id.unique()))
        user_u = np.array(sorted(self.original.user_id.unique()))

        if num_batches == 1:
            batches = user_u
        else:
            batches = np.array_split(user_u, num_batches)
        all_recom = dict()

        for i in range(num_batches):
            user_batch = batches[i]
            user_prog = self.original[self.original['user_id'].isin(user_batch)]
            user_prog = user_prog[['user_id', 'program_id']]
            recom = self.__recommend_batch__(user_batch, rows, cols, user_prog, np.array(item_u), k)
            all_recom.update(recom)
        self.model = all_recom
        return all_recom

    def recommend(self, user):
        if not self.programas_df:
            return self.all_recom[user]
        else:
            progs = self.all_recom[user]
        return self.programas_df[self.programas_df.isin(progs)]


def run():
    rec = Recomendador(data_dir='data/user_program_count.csv')
    rec.tipos_correctos()
    date = datetime.now() - relativedelta(years=2)
    rec.preprocess(date=date, k=10)
    U, sigma, V = rec.SVD()
    rec.recommend_whole(k=5, num_batches=20)
    return rec
