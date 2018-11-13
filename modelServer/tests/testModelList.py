import zmq
import sys
from protobuf.py.AlphaDotsModel_pb2 import ProtoModel
from protobuf.py.ModelList_pb2 import ModelListRequest, ModelList

connectAddr = "127.0.0.1:13452"
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://" + connectAddr)
print("get")
req = ModelListRequest()
req.action = ModelListRequest.GET
socket.send(req.SerializeToString())
resp = socket.recv()
print("resp:")
models = ModelList()
models.ParseFromString(resp)
for model in models.models:
    print(model.name)
#print(resp)

req2 = ModelListRequest()
req.action = ModelListRequest.ADD
req.model.name = 'testmodel123'
req.model.desc = 'test desc'
req.model.path = '/test/path/123.h5'
req.model.type = 'PolicyValue'
req.model.ai = 'ConvNet'
socket.send(req.SerializeToString())
resp = socket.recv()
print("resp:")
models = ModelList()
models.ParseFromString(resp)
for model in models.models:
    print(model.name)
