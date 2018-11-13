import zmq
import sys
from protobuf.py.ModelServer_pb2 import ModelServerRequest, ModelServerResponse
import numpy as np
from testUtils import *

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print(sys.getdefaultencoding())

get_status(12352)

# request a prediction
print("="*80)
print("prediction request...")
request = ModelServerRequest()
request.action = ModelServerRequest.PREDICT
request.predictionRequest.modelHandler = 'PolicyValue'
request.predictionRequest.modelKey = 'efg'
request.predictionRequest.image.width = 13
request.predictionRequest.image.height = 11
request.predictionRequest.image.pixels.extend([215 for _ in range(13*11)])
socket = open_socket(12355)
socket.send(request.SerializeToString())

print("prediction response...")
response = ModelServerResponse()
response.ParseFromString(socket.recv())
print_response(response)
print("resp.value: " + str(response.predictionResponse.pvdata.value))
npimg = np.array(response.predictionResponse.pvdata.policy)
print("resp.policy: " + str(npimg))


# request a prediction
print("="*80)
print("prediction request...")
request = ModelServerRequest()
request.action = ModelServerRequest.PREDICT
request.predictionRequest.modelHandler = 'PolicyValue'
request.predictionRequest.modelKey = 'cde'
request.predictionRequest.image.width = 13
request.predictionRequest.image.height = 15
request.predictionRequest.image.pixels.extend([215 for _ in range(13*15)])
socket = open_socket(12355)
socket.send(request.SerializeToString())

print("prediction response...")
response = ModelServerResponse()
response.ParseFromString(socket.recv())
print_response(response)
print("resp.value: " + str(response.predictionResponse.pvdata.value))
npimg = np.array(response.predictionResponse.pvdata.policy)
print("resp.policy: " + str(npimg))


# request a prediction
print("="*100)
print("prediction request...")
request = ModelServerRequest()
request.action = ModelServerRequest.PREDICT
request.predictionRequest.modelHandler = 'PolicyValue'
request.predictionRequest.modelKey = 'abc'
request.predictionRequest.image.width = 13
request.predictionRequest.image.height = 11
request.predictionRequest.image.pixels.extend([215 for _ in range(13*11)])
socket = open_socket(12354)
socket.send(request.SerializeToString())

print("prediction response...")
response = ModelServerResponse()
response.ParseFromString(socket.recv())
print_response(response)
print("resp.value: " + str(response.predictionResponse.pvdata.value))
npimg = np.array(response.predictionResponse.pvdata.policy)
print("resp.policy: " + str(npimg))