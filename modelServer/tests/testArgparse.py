import argparse

parser = argparse.ArgumentParser(description='Train the alpha zero model on a given dataset')
parser.add_argument('--dataset', action='append')
parser.add_argument('--no-augmentation', action='store_true')

args = parser.parse_args()

print(args)