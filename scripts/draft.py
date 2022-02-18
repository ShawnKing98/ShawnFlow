import matplotlib.pyplot as plt
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dist-metric', type=int, choices=['L2', 'abc', 3], default='bbb', help="the distance metric between two sets of trajectory")

    args = parser.parse_args()
    for (arg, value) in args._get_kwargs():
        print(f"{arg}: {value}, type: {type(value)}")
    return args

args = parse_arguments()
