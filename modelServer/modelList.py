# encoding=utf8
import os
import sys
import yaml
import zmq
import protobuf.py.AlphaDotsModel_pb2 as ProtoModel
from protobuf.py.ModelList_pb2 import ModelList, ModelListRequest
from Logger import Logger

logger = Logger(True, "/tmp/alphaDotsModelListServer.log", '[ModelList]')

listenAddr = "127.0.0.1:13452"
scriptDir = os.path.dirname(os.path.realpath(__file__))
modelListPath = scriptDir + "/models/models.yaml"

def readYAMLModelList():
    with open(modelListPath, 'r') as stream:
        try:
            modelData = yaml.load(stream)
            return modelData['models']

        except yaml.YAMLError as exc:
            print(exc)
            return []

def getProtobufModelList():
    protoModelList = ModelList()
    models = readYAMLModelList()
    logger.write("--- model list start ---")
    for model in models:
        logger.write(model['name'])
        protoModel = protoModelList.models.add()
        protoModel.name = model['name']
        protoModel.desc = model['desc']
        protoModel.path = model['path']
        protoModel.type = model['type']
        protoModel.ai = model['ai']
    logger.write("--- model list end ---")

    return protoModelList.SerializeToString()

class Model:
    def __init__(self, name, desc, path, type, ai, options=None):
        self.name = name
        self.desc = desc
        self.path = path
        self.type = type
        self.ai = ai
        self.options = options

def saveModelToYAML(newModel):
    newModelD = {'name': str(newModel.name),
                 'desc': str(newModel.desc),
                 'path': str(newModel.path),
                 'type': str(newModel.type),
                 'ai': str(newModel.ai)}
    models = readYAMLModelList()
    found_model = False
    for model_idx in range(len(models)):
        model = models[model_idx]
        if model['name'] == newModel.name:
            logger.write("model already exists, replacing model")
            models[model_idx] = newModelD
            found_model = True
    if not found_model:
        models.append(newModelD)
    with open("models/models.yaml", "w") as modelfile:
        yaml.dump({'models': models}, modelfile, default_flow_style=False)
#    with open("models/models.yaml", "a") as modelyaml:
#        modelyaml.write("\n")
#        modelyaml.write("  - name: " + model.name + "\n")
#        modelyaml.write("    desc: " + model.desc + "\n")
#        modelyaml.write("    path: " + model.path + "\n")
#        modelyaml.write("    type: " + model.type + "\n")
#        modelyaml.write("    ai: " + model.ai + "\n")
#        if model.options is not None:
#            modelyaml.write("    options: " + model.options + "\n")


if __name__ == "__main__":
    logger.write("binding tcp socket to %s" % listenAddr)
    sys.stdout.flush()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://" + listenAddr)

    # prepare list for get request
    serializedList = getProtobufModelList()

    # handle requests
    while True:
        message = socket.recv()
        request = ModelListRequest()
        request.ParseFromString(message)
        if request.action == ModelListRequest.GET:
            socket.send(serializedList)
        if request.action == ModelListRequest.ADD:
            if not request.HasField("model"):
                logger.write("ERROR: ADD request without model!")
            else:
                logger.write("ADD model")
                saveModelToYAML(request.model)
            serializedList = getProtobufModelList()
            socket.send(serializedList)
        if request.action == ModelListRequest.REMOVE:
            # TODO: remove model!
            logger.write("REMOVE not implemented yet!")
            socket.send(serializedList)
