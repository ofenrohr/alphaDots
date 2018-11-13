import zmq
from protobuf.py.ModelServer_pb2 import ModelServerRequest, ModelServerResponse

def open_socket(port):
    connectAddr = "127.0.0.1:" + str(port)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://" + connectAddr)
    return socket

def print_response(response):
    status_dict = {ModelServerResponse.RESP_OK: 'RESP_OK',
                   ModelServerResponse.RESP_FAIL: 'RESP_FAIL'}
    print("resp.status = " + status_dict[response.status])
    print("resp.errorMessage = " + response.errorMessage)

def get_status(port):
    socket = open_socket(port)
    req = ModelServerRequest()
    req.action = ModelServerRequest.STATUS
    socket.send(req.SerializeToString())
    resp = ModelServerResponse()
    resp.ParseFromString(socket.recv())
    print("STATUS:")
    print(resp.statusMessage)
    print("\n")
