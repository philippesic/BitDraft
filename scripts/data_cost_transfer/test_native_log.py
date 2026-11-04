import unittest,tempfile
from pathlib import Path
import native_log_text
class NativeLogTest(unittest.TestCase):
 def test_lossless_only_for_native_log(self):
  original=Path.read_text
  try:
   native_log_text.install()
   with tempfile.TemporaryDirectory() as root:
    root=Path(root);raw=b'candidate token=42 piece=\xf0\x9f\nnext token=11\n'
    p=root/'server.log';p.write_bytes(raw);text=p.read_text()
    self.assertEqual(text.encode('utf-8','surrogateescape'),raw)
    q=root/'response.json';q.write_bytes(raw)
    with self.assertRaises(UnicodeDecodeError):q.read_text()
    with self.assertRaises(UnicodeDecodeError):p.read_text(errors='strict')
  finally:Path.read_text=original
if __name__=='__main__':unittest.main()
