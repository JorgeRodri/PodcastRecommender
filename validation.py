import json
import csv
from MySQL.connection import getConnection
import pandas as pd
from pymysql import err
import warnings


def read_recom(file):
    d = dict()
    with open(file, 'r') as f:
        for line in f:
            if 'row' in line:
                c = line.replace('"', '').rstrip()
                o = c.split(', ')
                d[o[1]] = o[2:]
            else:
                c = line.replace('"', '').rstrip()
                o = c.split(', ')
                d[o[0]] = o[1:]
    return d


def __get_programs_connection_error__():
    programs = pd.read_csv('data/Programs_from_2017.csv', header=0, error_bad_lines=False, warn_bad_lines=False)
    programs.columns = ['id', 'name', 'recomendations', 'category', 'subcat', 'audios', 'last_update']
    programs = programs[['id', 'name']]
    programs['id'] = programs['id'].astype(str)
    programs['name'] = programs['name'].astype(str)
    return programs


def get_programs():
    """
    function to get the table with the programs from the MySQL server.
    :return: DataFrame containing the programs and their information.
    """
    with open('C:/Users/jorge.rodriguez/PycharmProjects/iVooxPython/MySQL/credenciales_corrector.json', 'r') as f:
        arch = json.loads(f.read())
    connection = getConnection(arch)
    query = "SELECT programs_id, programs_name FROM ivoox.programs"
    df = pd.read_sql(query, con=connection)

    df.columns = ['id', 'name']

    df['id'] = df['id'].astype(str)
    df['name'] = df['name'].astype(str)
    return df


print('Validar los resultados. Obteniendo el recomendador.')
my_file = 'TEsts/recommender_jorge_200_10.csv'
r = read_recom(my_file)


print('Descargando programas.')
try:
    programs = get_programs()
except err.InternalError as e:
    print(e)
    warnings.warn('Fail to conect to the sever, using the downloaded programs dataframe, some programs maybe missing.')
    programs = __get_programs_connection_error__()

except err.OperationalError as e2:
    print(e2)
    warnings.warn('Fail to connect to the sever, using the downloaded programs dataframe, some programs maybe missing.')
    programs = __get_programs_connection_error__()


ivoox = {'Jose1': '5209511', 'Fede': '19764', 'Juan0': '2982917', 'Juan1': '7811817', 'Miguel': '911419',
         'Emilio': '1276315', 'Yo': '6776060', 'Laura': '138931'}


if __name__ == '__main__':
    print(list(ivoox.keys()))
    while True:
        user = str(input('Choose a user ID: '))
        if user == 'No' or user == 'Exit' or user == 'exit' or user == '' or user == 'quit' or user == 'Quit':
            print('Exiting...')
            break
        elif not user.isdigit():
            try:
                user = ivoox[user]
            except KeyError:
                print('User not in recommender. If you want to finish write: NO, Exit, exit, "", quit or Quit.\n')
                continue
        try:
            recoms = r[user]
            print(recoms)
            user_recommendation = programs[programs.id.isin(recoms)]
            user_recommendation.loc[:, 'id'] = user_recommendation['id'].astype("category")
            user_recommendation.id.cat.set_categories(recoms, inplace=True)
            print(user_recommendation.sort_values("id"))
            print()
        except KeyError:
            print('User not in recommender. If you want to finish write: NO, Exit, exit, "", quit or Quit.\n')
