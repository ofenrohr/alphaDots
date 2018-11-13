# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ModelServer.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import ProcessManagement_pb2 as ProcessManagement__pb2
import Prediction_pb2 as Prediction__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='ModelServer.proto',
  package='AlphaDots',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x11ModelServer.proto\x12\tAlphaDots\x1a\x17ProcessManagement.proto\x1a\x10Prediction.proto\"\xf8\x01\n\x12ModelServerRequest\x12:\n\x06\x61\x63tion\x18\x01 \x02(\x0e\x32*.AlphaDots.ModelServerRequest.ServerAction\x12\x38\n\x0bmgmtRequest\x18\x02 \x01(\x0b\x32#.AlphaDots.ProcessManagementRequest\x12\x37\n\x11predictionRequest\x18\x03 \x01(\x0b\x32\x1c.AlphaDots.PredictionRequest\"3\n\x0cServerAction\x12\n\n\x06MANAGE\x10\x01\x12\x0b\n\x07PREDICT\x10\x02\x12\n\n\x06STATUS\x10\x03\"\xa6\x02\n\x13ModelServerResponse\x12=\n\x06status\x18\x01 \x02(\x0e\x32-.AlphaDots.ModelServerResponse.ServerResponse\x12\x14\n\x0c\x65rrorMessage\x18\x02 \x01(\t\x12:\n\x0cmgmtResponse\x18\x03 \x01(\x0b\x32$.AlphaDots.ProcessManagementResponse\x12\x39\n\x12predictionResponse\x18\x04 \x01(\x0b\x32\x1d.AlphaDots.PredictionResponse\x12\x15\n\rstatusMessage\x18\x05 \x01(\t\",\n\x0eServerResponse\x12\x0b\n\x07RESP_OK\x10\x01\x12\r\n\tRESP_FAIL\x10\x02')
  ,
  dependencies=[ProcessManagement__pb2.DESCRIPTOR,Prediction__pb2.DESCRIPTOR,])



_MODELSERVERREQUEST_SERVERACTION = _descriptor.EnumDescriptor(
  name='ServerAction',
  full_name='AlphaDots.ModelServerRequest.ServerAction',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MANAGE', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PREDICT', index=1, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STATUS', index=2, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=273,
  serialized_end=324,
)
_sym_db.RegisterEnumDescriptor(_MODELSERVERREQUEST_SERVERACTION)

_MODELSERVERRESPONSE_SERVERRESPONSE = _descriptor.EnumDescriptor(
  name='ServerResponse',
  full_name='AlphaDots.ModelServerResponse.ServerResponse',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='RESP_OK', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RESP_FAIL', index=1, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=577,
  serialized_end=621,
)
_sym_db.RegisterEnumDescriptor(_MODELSERVERRESPONSE_SERVERRESPONSE)


_MODELSERVERREQUEST = _descriptor.Descriptor(
  name='ModelServerRequest',
  full_name='AlphaDots.ModelServerRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='action', full_name='AlphaDots.ModelServerRequest.action', index=0,
      number=1, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mgmtRequest', full_name='AlphaDots.ModelServerRequest.mgmtRequest', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='predictionRequest', full_name='AlphaDots.ModelServerRequest.predictionRequest', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _MODELSERVERREQUEST_SERVERACTION,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=76,
  serialized_end=324,
)


_MODELSERVERRESPONSE = _descriptor.Descriptor(
  name='ModelServerResponse',
  full_name='AlphaDots.ModelServerResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='AlphaDots.ModelServerResponse.status', index=0,
      number=1, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='errorMessage', full_name='AlphaDots.ModelServerResponse.errorMessage', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mgmtResponse', full_name='AlphaDots.ModelServerResponse.mgmtResponse', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='predictionResponse', full_name='AlphaDots.ModelServerResponse.predictionResponse', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='statusMessage', full_name='AlphaDots.ModelServerResponse.statusMessage', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _MODELSERVERRESPONSE_SERVERRESPONSE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=327,
  serialized_end=621,
)

_MODELSERVERREQUEST.fields_by_name['action'].enum_type = _MODELSERVERREQUEST_SERVERACTION
_MODELSERVERREQUEST.fields_by_name['mgmtRequest'].message_type = ProcessManagement__pb2._PROCESSMANAGEMENTREQUEST
_MODELSERVERREQUEST.fields_by_name['predictionRequest'].message_type = Prediction__pb2._PREDICTIONREQUEST
_MODELSERVERREQUEST_SERVERACTION.containing_type = _MODELSERVERREQUEST
_MODELSERVERRESPONSE.fields_by_name['status'].enum_type = _MODELSERVERRESPONSE_SERVERRESPONSE
_MODELSERVERRESPONSE.fields_by_name['mgmtResponse'].message_type = ProcessManagement__pb2._PROCESSMANAGEMENTRESPONSE
_MODELSERVERRESPONSE.fields_by_name['predictionResponse'].message_type = Prediction__pb2._PREDICTIONRESPONSE
_MODELSERVERRESPONSE_SERVERRESPONSE.containing_type = _MODELSERVERRESPONSE
DESCRIPTOR.message_types_by_name['ModelServerRequest'] = _MODELSERVERREQUEST
DESCRIPTOR.message_types_by_name['ModelServerResponse'] = _MODELSERVERRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ModelServerRequest = _reflection.GeneratedProtocolMessageType('ModelServerRequest', (_message.Message,), dict(
  DESCRIPTOR = _MODELSERVERREQUEST,
  __module__ = 'ModelServer_pb2'
  # @@protoc_insertion_point(class_scope:AlphaDots.ModelServerRequest)
  ))
_sym_db.RegisterMessage(ModelServerRequest)

ModelServerResponse = _reflection.GeneratedProtocolMessageType('ModelServerResponse', (_message.Message,), dict(
  DESCRIPTOR = _MODELSERVERRESPONSE,
  __module__ = 'ModelServer_pb2'
  # @@protoc_insertion_point(class_scope:AlphaDots.ModelServerResponse)
  ))
_sym_db.RegisterMessage(ModelServerResponse)


# @@protoc_insertion_point(module_scope)