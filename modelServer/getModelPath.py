from modelList import readYAMLModelList
import sys

for model in readYAMLModelList():
    if model['name'] == sys.argv[1]:
        print(model['path'])

exit(0)

