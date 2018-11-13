import zmq
import sys
sys.path.append('..')
from protobuf.py.ProcessManagement_pb2 import ProcessManagementRequest, ProcessManagementResponse

print(sys.getdefaultencoding())
connectAddr = "127.0.0.1:12353"
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://" + connectAddr)

print("request...")
req = ProcessManagementRequest()
req.model = "AlphaZeroV14"
req.width = 13
req.height = 11
req.key = "abc"
req.action = ProcessManagementRequest.START
req.gpu = False
socket.send(req.SerializeToString())

print("response...")
resp = ProcessManagementResponse()
resp.ParseFromString(socket.recv())
print("resp.key: "+resp.key)
print("resp.port: "+str(resp.port))
#print(resp)