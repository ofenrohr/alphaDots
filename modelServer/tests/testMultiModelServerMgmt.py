import zmq
import sys
from protobuf.py.ProcessManagement_pb2 import ProcessManagementRequest, ProcessManagementResponse
from protobuf.py.ModelServer_pb2 import ModelServerRequest, ModelServerResponse
from testUtils import *

print(sys.getdefaultencoding())
connectAddr = "127.0.0.1:12352"
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://" + connectAddr)

get_status(12352)

# Start a process without gpu...
print("management request...")
mgmt_request = ProcessManagementRequest()
mgmt_request.model = "AlphaZeroV14"
mgmt_request.width = 5
mgmt_request.height = 4
mgmt_request.key = "abc"
mgmt_request.action = ProcessManagementRequest.START
mgmt_request.gpu = False

request = ModelServerRequest()
request.action = ModelServerRequest.MANAGE
request.mgmtRequest.MergeFrom(mgmt_request)
socket.send(request.SerializeToString())

print("management response...")
response = ModelServerResponse()
response.ParseFromString(socket.recv())
mgmt_response = response.mgmtResponse
print("resp.key: " + mgmt_response.key)
print("resp.port: " + str(mgmt_response.port))


get_status(12352)

# start a process with gpu...
print("management request...")
mgmt_request = ProcessManagementRequest()
mgmt_request.model = "AlphaZeroV7"
mgmt_request.width = 5
mgmt_request.height = 6
mgmt_request.key = "cde"
mgmt_request.action = ProcessManagementRequest.START
mgmt_request.gpu = True

request = ModelServerRequest()
request.action = ModelServerRequest.MANAGE
request.mgmtRequest.MergeFrom(mgmt_request)
socket.send(request.SerializeToString())

print("management response...")
response = ModelServerResponse()
response.ParseFromString(socket.recv())
mgmt_response = response.mgmtResponse
print("resp.key: " + mgmt_response.key)
print("resp.port: " + str(mgmt_response.port))


# start another process with gpu...
print("management request...")
mgmt_request = ProcessManagementRequest()
mgmt_request.model = "AlphaZeroV11"
mgmt_request.width = 5
mgmt_request.height = 4
mgmt_request.key = "efg"
mgmt_request.action = ProcessManagementRequest.START
mgmt_request.gpu = True

request = ModelServerRequest()
request.action = ModelServerRequest.MANAGE
request.mgmtRequest.MergeFrom(mgmt_request)
socket.send(request.SerializeToString())

print("management response...")
response = ModelServerResponse()
response.ParseFromString(socket.recv())
mgmt_response = response.mgmtResponse
print("resp.key: " + mgmt_response.key)
print("resp.port: " + str(mgmt_response.port))


get_status(12352)

# start another process
print("management request...")
mgmt_request = ProcessManagementRequest()
mgmt_request.model = "AlphaZeroV13"
mgmt_request.width = 4
mgmt_request.height = 4
mgmt_request.key = "ghi"
mgmt_request.action = ProcessManagementRequest.START
mgmt_request.gpu = False

request = ModelServerRequest()
request.action = ModelServerRequest.MANAGE
request.mgmtRequest.MergeFrom(mgmt_request)
socket.send(request.SerializeToString())

print("management response...")
response = ModelServerResponse()
response.ParseFromString(socket.recv())
mgmt_response = response.mgmtResponse
print("resp.key: " + mgmt_response.key)
print("resp.port: " + str(mgmt_response.port))

get_status(12352)

# stop the process
print("management request...")
mgmt_request = ProcessManagementRequest()
mgmt_request.model = "AlphaZeroV13"
mgmt_request.width = 4
mgmt_request.height = 4
mgmt_request.key = "ghi"
mgmt_request.action = ProcessManagementRequest.STOP
mgmt_request.gpu = False

request = ModelServerRequest()
request.action = ModelServerRequest.MANAGE
request.mgmtRequest.MergeFrom(mgmt_request)
socket.send(request.SerializeToString())

print("management response...")
response = ModelServerResponse()
response.ParseFromString(socket.recv())
mgmt_response = response.mgmtResponse
print("resp.key: " + mgmt_response.key)
print("resp.port: " + str(mgmt_response.port))


get_status(12352)


# start another process
print("management request...")
mgmt_request = ProcessManagementRequest()
mgmt_request.model = "AlphaZeroV13"
mgmt_request.width = 4
mgmt_request.height = 4
mgmt_request.key = "ghi"
mgmt_request.action = ProcessManagementRequest.START
mgmt_request.gpu = False

request = ModelServerRequest()
request.action = ModelServerRequest.MANAGE
request.mgmtRequest.MergeFrom(mgmt_request)
socket.send(request.SerializeToString())

print("management response...")
response = ModelServerResponse()
response.ParseFromString(socket.recv())
mgmt_response = response.mgmtResponse
print("resp.key: " + mgmt_response.key)
print("resp.port: " + str(mgmt_response.port))

get_status(12352)

