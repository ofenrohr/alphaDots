import yaml
import sys

from modelList import readYAMLModelList, Model

models = readYAMLModelList()

print(models)

yaml.dump({'models': models}, sys.stdout, default_flow_style=False)


