
def read_recom(file):
    d = dict()
    with open(file, 'r') as f:
        for line in f:
            c = line.replace('"', '').rstrip()
            o = c.split(', ')
            d[o[0]] = o[1:]
    return d


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


if __name__ == '__main__':
    my_file = 'recommender_jorge_200.csv'
    r = read_recom(my_file)
    to_csv(r, my_file)

