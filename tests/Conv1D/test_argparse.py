import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--variable', help='help message', type=float)
args = parser.parse_args()

print('Valuer of variable= {}'.format(args.variable))