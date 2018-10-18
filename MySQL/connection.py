import pymysql.cursors


# Function return a connection.
def getConnection(credenciales):
    """
    parameters already fixed inside the function
    :return: the connection to the server
    """
    connection = pymysql.connect(charset='utf8',
                                 cursorclass=pymysql.cursors.DictCursor,
                                 **credenciales)
    return connection


