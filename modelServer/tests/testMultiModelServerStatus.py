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

