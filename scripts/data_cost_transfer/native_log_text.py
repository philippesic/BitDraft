"""Lossless UTF-8 decoding for native logs containing partial token byte pieces.

HTTP JSON remains strict UTF-8. Numeric trace fields remain unchanged. Invalid
bytes round-trip through surrogateescape; no byte is replaced or discarded.
"""
from pathlib import Path
_original=Path.read_text

def install():
 def read_text(path,encoding=None,errors=None):
  if path.name!='server.log' or errors is not None:return _original(path,encoding=encoding,errors=errors)
  raw=path.read_bytes();text=raw.decode(encoding or 'utf-8',errors='surrogateescape')
  if text.encode(encoding or 'utf-8',errors='surrogateescape')!=raw:raise ValueError('native log byte round-trip failed')
  return text
 Path.read_text=read_text
