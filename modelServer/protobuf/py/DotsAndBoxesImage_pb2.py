# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: DotsAndBoxesImage.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='DotsAndBoxesImage.proto',
  package='AlphaDots',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x17\x44otsAndBoxesImage.proto\x12\tAlphaDots\"s\n\x11\x44otsAndBoxesImage\x12\r\n\x05width\x18\x01 \x02(\x05\x12\x0e\n\x06height\x18\x02 \x02(\x05\x12\x0e\n\x06pixels\x18\x03 \x03(\x05\x12/\n\tnextImage\x18\x04 \x01(\x0b\x32\x1c.AlphaDots.DotsAndBoxesImage')
)




_DOTSANDBOXESIMAGE = _descriptor.Descriptor(
  name='DotsAndBoxesImage',
  full_name='AlphaDots.DotsAndBoxesImage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='width', full_name='AlphaDots.DotsAndBoxesImage.width', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='AlphaDots.DotsAndBoxesImage.height', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pixels', full_name='AlphaDots.DotsAndBoxesImage.pixels', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nextImage', full_name='AlphaDots.DotsAndBoxesImage.nextImage', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=38,
  serialized_end=153,
)

_DOTSANDBOXESIMAGE.fields_by_name['nextImage'].message_type = _DOTSANDBOXESIMAGE
DESCRIPTOR.message_types_by_name['DotsAndBoxesImage'] = _DOTSANDBOXESIMAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DotsAndBoxesImage = _reflection.GeneratedProtocolMessageType('DotsAndBoxesImage', (_message.Message,), dict(
  DESCRIPTOR = _DOTSANDBOXESIMAGE,
  __module__ = 'DotsAndBoxesImage_pb2'
  # @@protoc_insertion_point(class_scope:AlphaDots.DotsAndBoxesImage)
  ))
_sym_db.RegisterMessage(DotsAndBoxesImage)


# @@protoc_insertion_point(module_scope)
