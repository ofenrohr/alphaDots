import argparse
import sys
from modelList import readYAMLModelList, Model as AlphaDotsModel, saveModelToYAML


# parse command line args
parser = argparse.ArgumentParser(description='Create a new AlphaZero model')
parser.add_argument('--name', required=True)
parser.add_argument('--newname', required=True)
parser.add_argument('--mcts', action='store_true')
args = parser.parse_args()

model_list = readYAMLModelList()

model_copy = None
for model in model_list:
    if model['name'] == args.name:
        if args.mcts:
            ai = 'MCTS-AlphaZero'
        else:
            ai = 'ConvNet'
        model_copy = AlphaDotsModel(args.newname, model['desc'], model['path'], model['type'], ai)
if model_copy is not None:
    print('adding model copy')
    saveModelToYAML(model_copy)
else:
    print('failed to find model named ' + args.name)
