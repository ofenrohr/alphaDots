# encoding=utf8
import os
import sys
import zmq
import argparse
import numpy as np

from protobuf.py.ProcessManagement_pb2 import ProcessManagementRequest, ProcessManagementResponse
from protobuf.py.Prediction_pb2 import PredictionRequest, PredictionResponse
from protobuf.py.ModelServer_pb2 import ModelServerRequest, ModelServerResponse
from multiprocessing import Process
from Logger import Logger


class MultiModelServer:
    """MultiModelServer
    Used to launch and run models (neural networks) for KSquares.
    """
    def __init__(self, is_master, port, debug, logdest, logger):
        # keep track of running models
        self.modelDict = {}
        self.is_active = True
        # zmq stuff
        self.context = None
        self.socket = None
        # config options
        self.is_master = is_master
        self.own_port = port
        self.debug = debug
        self.logdest = logdest
        self.logger = logger

        # only used in master mode
        self.port = self.own_port + 1

        # setup zeromq
        self.bind_management_port()

        # helper stuff
        self.status_dict = {ModelServerResponse.RESP_OK: 'RESP_OK',
                            ModelServerResponse.RESP_FAIL: 'RESP_FAIL'}
        self.server_action_dict = {ModelServerRequest.MANAGE: 'MANAGE',
                                   ModelServerRequest.PREDICT: 'PREDICT',
                                   ModelServerRequest.STATUS: 'STATUS'}
        self.action_dict = {ProcessManagementRequest.START: 'START',
                            ProcessManagementRequest.STOP: 'STOP'}
        # create ModelServerResponse and reuse it
        self.response = ModelServerResponse()

    @staticmethod
    def run(is_master, port, debug, logdest):
        """ Create an instance of MultiModelServer and run it """

        # setup logging
        logger = Logger(debug, logdest + "/alphaDotsModelServer"+str(port)+".log", '['+str(port)+']')
        sys.stderr = Logger(debug, logdest + "/alphaDotsModelServer"+str(port)+".err", '[E]['+str(port)+']')

        np.set_printoptions(precision=2, suppress=True, linewidth=90)

        multiModelServer = MultiModelServer(is_master, port, debug, logdest, logger)
        while multiModelServer.is_active:
            multiModelServer.handle_request()

    def bind_management_port(self):
        # bind zmq socket
        listenAddr = "127.0.0.1:" + str(self.own_port)
        self.logger.write("binding zeromq socket to tcp://%s" % listenAddr)
        sys.stdout.flush()
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://" + listenAddr)

    def zmq_request(self, port, request):
        connectAddr = "127.0.0.1:" + str(port)
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://" + connectAddr)
        socket.send(request)
        return socket.recv()

    def launch_model_server(self, mgmt_request):
        """Launch a model server"""

        gpu = mgmt_request.gpu

        self.port = self.port + 1

        # disable gpu if requested
        if not gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        os.environ['PYTHONPATH'] = '.:protobuf/py'

        # start the process
        p = Process(target=MultiModelServer.run,
                    args=(False, self.port, self.debug, self.logdest))
        p.daemon = True
        p.start()

        # reset gpu disabling environment flag
        if not gpu:
            del os.environ['CUDA_VISIBLE_DEVICES']

        return p

    def load_alphadots_model(self, model, width, height):
        if self.is_master:
            self.logger.write("ERROR: master mode model server may not load models!")
            return None

        # only import keras based modules when we really need them
        from keras.models import load_model
        import modelList
        import modelHandlers.DirectInference
        import modelHandlers.Sequence
        import modelHandlers.DirectInferenceCategorical
        import modelHandlers.SequenceCategorical
        import modelHandlers.PolicyValue
        from models.LineFilterLayer import LineFilterLayer
        from models.SequenceLineFilterLayer import SequenceLineFilterLayer
        from models.ValueLayer import ValueLayer

        # load model
        self.logger.write("[LOAD MODEL] preparing to load model " + model)
        sys.stdout.flush()

        # LineFilterLayer has to be set before loading the model
        LineFilterLayer.imgWidth = width
        LineFilterLayer.imgHeight = height
        # SequenceLineFilterLayer has to be set before loading the model
        SequenceLineFilterLayer.imgWidth = width
        SequenceLineFilterLayer.imgHeight = height
        # ValueLayer has to be set before loading the model
        ValueLayer.imgWidth = width
        ValueLayer.imgHeight = height

        self.logger.write("[LOAD MODEL] reading model list")
        sys.stdout.flush()

        models = modelList.readYAMLModelList()
        assert (len(models) > 0)

        selectedModel = models[0]
        foundModel = False
        for m in models:
            if m['name'] == model:
                selectedModel = m
                foundModel = True

        if not foundModel:
            self.logger.write("ERROR: Can't find model!")
            self.logger.write("models:")
            self.logger.write(models)
            exit(1)

        self.logger.write("[LOAD MODEL] loading model: %s" % selectedModel)
        sys.stdout.flush()

        scriptDir = os.path.dirname(os.path.realpath(__file__))
        model = load_model(scriptDir + "/models/" + selectedModel['path'],
                           custom_objects={'LineFilterLayer': LineFilterLayer,
                                           'SequenceLineFilterLayer': SequenceLineFilterLayer,
                                           'ValueLayer': ValueLayer})

        self.logger.write("[LOAD MODEL] preparing model handler")
        sys.stdout.flush()

        # load model handler
        modelHandler = None
        if selectedModel['type'] == 'DirectInference':
            modelHandler = modelHandlers.DirectInference.DirectInference(model, self.debug)
        elif selectedModel['type'] == 'Sequence':
            modelHandler = modelHandlers.Sequence.Sequence(model)
        elif selectedModel['type'] == 'DirectInferenceCategorical':
            modelHandler = modelHandlers.DirectInferenceCategorical.DirectInferenceCategorical(model)
        elif selectedModel['type'] == 'SequenceCategorical':
            modelHandler = modelHandlers.SequenceCategorical.SequenceCategorical(model)
        elif selectedModel['type'] == 'PolicyValue':
            invertValue = False
            if 'options' in selectedModel and 'InvertValue' in selectedModel['options']:
                invertValue = True
            modelHandler = modelHandlers.PolicyValue.PolicyValue(model, invertValue, self.debug, self.logdest,
                                                                 self.logger)
        else:
            self.logger.write("unknown model handler type!")
            exit(1)

        self.logger.write("[LOAD MODEL] starting model handler %s" % selectedModel['type'])
        return modelHandler

    def handle_mgmt_request(self, mgmt_request):
        """
        Handles management requests. Either starts a new process or loads the model in this process if it's GPU enabled
        :param mgmt_request: protobuf ManagementRequest
        :return: protobuf ManagementResponse
        """
        self.logger.write("received ProcessManagementRequest:")
        self.logger.write(" -> model: " + mgmt_request.model)
        self.logger.write(" -> width: " + str(mgmt_request.width))
        self.logger.write(" -> height: " + str(mgmt_request.height))
        self.logger.write(" -> key: " + mgmt_request.key)
        self.logger.write(" -> action: " + self.action_dict[mgmt_request.action])
        self.logger.write(" -> gpu: " + str(mgmt_request.gpu))
        sys.stdout.flush()

        status = True
        status_msg = ""
        if mgmt_request.action == ProcessManagementRequest.START:
            self.logger.write("action: START")
            sys.stdout.flush()

            if self.is_master:
                if mgmt_request.gpu:
                    model_process = None
                    model_port = -1
                    # search for model process that has gpu enabled
                    for model_key in self.modelDict:
                        if self.modelDict[model_key]['gpu']:
                            model_process = self.modelDict[model_key]['process']
                            model_port = self.modelDict[model_key]['port']
                    # no gpu process -> start a gpu process
                    if model_process is None or not model_process.is_alive():
                        self.logger.write("launching gpu enabled model server")
                        model_process = self.launch_model_server(mgmt_request)
                        model_port = self.port
                else:
                    model_process = self.launch_model_server(mgmt_request)
                    model_port = self.port

                self.modelDict[mgmt_request.key] = {'external_process': True, 'process': model_process, 'port': model_port,
                                                    'gpu': mgmt_request.gpu}

                # send start model request to process
                self.logger.write("Sending load model request to process")
                load_model_request = ModelServerRequest()
                load_model_request.action = ModelServerRequest.MANAGE
                load_model_request.mgmtRequest.MergeFrom(mgmt_request)
                load_model_response = ModelServerResponse()
                load_model_response.ParseFromString(self.zmq_request(self.port, load_model_request.SerializeToString()))
                self.logger.write("Load model response: " + self.status_dict[load_model_response.status])
            else:
                model = mgmt_request.model
                width = mgmt_request.width * 2 + 3
                height = mgmt_request.height * 2 + 3
                handler = self.load_alphadots_model(model, width, height)
                self.modelDict[mgmt_request.key] = {'external_process': False, 'handler': handler,
                                                    'port': self.own_port, 'gpu': mgmt_request.gpu}
        else:
            self.logger.write("action: STOP")
            sys.stdout.flush()


            if self.is_master:
                if mgmt_request.model == "":
                    self.is_active = False
                else:
                    stop_server_request = ModelServerRequest()
                    stop_server_request.action = ModelServerRequest.MANAGE
                    stop_server_request.mgmtRequest.MergeFrom(mgmt_request)
                    stop_server_response = ModelServerResponse()
                    stop_server_response.ParseFromString(
                        self.zmq_request(self.port, stop_server_request.SerializeToString()))
                    self.logger.write("Stop response: "+str(stop_server_response.status))
            else:
                self.logger.write("unloading model " + mgmt_request.key)
                del self.modelDict[mgmt_request.key]
                if len(self.modelDict) == 0:
                    self.logger.write("stopping server process...")
                    self.is_active = False

        response = ModelServerResponse()
        if mgmt_request.key in self.modelDict:
            response.mgmtResponse.port = self.modelDict[mgmt_request.key]['port']
        else:
            response.mgmtResponse.port = -1
        response.mgmtResponse.key = mgmt_request.key
        if status:
            response.status = ModelServerResponse.RESP_OK
        else:
            response.status = ModelServerResponse.RESP_FAIL
            response.errorMessage = status_msg

        self.logger.write("Sending management response: ")
        self.logger.write("resp.status: "+self.status_dict[response.status])
        self.logger.write("resp.errorMessage: "+response.errorMessage)
        self.logger.write("resp.statusMessage: "+response.statusMessage)
        return response.SerializeToString()

    def handle_prediction_request(self, predictionRequest):
        if self.debug:
            self.logger.write("model server prediction request for \"" + predictionRequest.modelKey +"\"")
        if not predictionRequest.modelKey in self.modelDict:
            response = ModelServerResponse()
            response.status = ModelServerResponse.RESP_FAIL
            response.errorMessage = "ERROR: unknown model key!"
            zmq_response = response.SerializeToString()
        elif self.modelDict[predictionRequest.modelKey]['external_process']:
            self.logger.write("ERROR: received prediction request for external process!")
            response = ModelServerResponse()
            response.status = ModelServerResponse.RESP_FAIL
            response.errorMessage = "ERROR: received prediction request for external process!"
            zmq_response = response.SerializeToString()
        else:
            handler = self.modelDict[predictionRequest.modelKey]['handler']
            if predictionRequest.HasField('categorical') and \
               predictionRequest.modelHandler == 'PolicyValue':
                handler.setCategorical(predictionRequest.categorical)
            response = ModelServerResponse()
            if predictionRequest.modelHandler in ['Sequence', 'SequenceCategorical']:
                prediction = handler.predict(predictionRequest.sequence)
            else:
                prediction = handler.predict(predictionRequest.image)
            if predictionRequest.modelHandler in ['PolicyValue']:
                response.predictionResponse.pvdata.MergeFrom(prediction)
            else:
                response.predictionResponse.image.MergeFrom(prediction)
            response.status = ModelServerResponse.RESP_OK
            response.errorMessage = ""
            response.statusMessage = ""
            zmq_response = response.SerializeToString()
        return zmq_response

    def handle_status_request(self):
        status_str = "is_master: " + str(self.is_master) + "\nport: " + str(self.own_port) + "\n"
        for model_key in self.modelDict:
            status_str += "="*80
            status_str += "\nmodel key: " + model_key
            status_str += "\nexternal process: " + str(self.modelDict[model_key]['external_process'])
            status_str += "\nport: " + str(self.modelDict[model_key]['port'])
            status_str += "\ngpu: " + str(self.modelDict[model_key]['gpu'])
            status_str += "\n"

        response = ModelServerResponse()
        response.status = ModelServerResponse.RESP_OK
        response.statusMessage = status_str
        return response.SerializeToString()

    def handle_request(self):
        request_str = self.socket.recv()

        request = ModelServerRequest()
        request.ParseFromString(request_str)

        if self.debug:
            self.logger.write("model server request: " + self.server_action_dict[request.action])

        zmq_response = ""

        # handle management request
        if request.action == ModelServerRequest.MANAGE:
            if self.debug:
                self.logger.write("model server manage request")

            zmq_response = self.handle_mgmt_request(request.mgmtRequest)

        # handle prediction request
        if request.action == ModelServerRequest.PREDICT:
            zmq_response = self.handle_prediction_request(request.predictionRequest)

        if request.action == ModelServerRequest.STATUS:
            zmq_response = self.handle_status_request()

        self.socket.send(zmq_response)


if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description='Provide model hosting for Dots and Boxes models')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--logdest', default='/tmp')
    args = parser.parse_args()

    # launch_model_server(args.model, args.width, args.height, args.port, args.debug, args.logdest)

    port = 12352

    MultiModelServer.run(True, port, args.debug, args.logdest)
