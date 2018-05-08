import argparse 

def get_args():
    parser = argparse.ArgumentParser(description='k-means clustering')
    parser.add_argument('-f', '--filename', help='.csv data file', required=True)
    parser.add_argument('-k', '--k', help='number of clusters', required=True)

    return vars(parser.parse_args())


def main():
    args = get_args()
    file_name = args['filename']
    k = args['k']

main()
