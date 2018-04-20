import pymysql.cursors


# Function return a connection.
def getConnection():
    """
    parameters already fixed inside the function
    :return: the connection to the server
    """
    host_ivoox = ''
    user_ivoox = ''
    password_ivoox = ''
    db_ivoox = ''

    connection = pymysql.connect(host=host_ivoox,
                                 user=user_ivoox,
                                 password=password_ivoox,
                                 db=db_ivoox,
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection


