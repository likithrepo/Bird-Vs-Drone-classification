import io
import json
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from PIL import Image
from radar import CNN
from app import handler

class AppTests(unittest.TestCase):
    def test_upload_flow_and_invalid_image(self):
        model = CNN(); model.threshold = .5; model.synthetic = True
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(model, {}))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(base) as response:
                self.assertIn(b'RadarScope', response.read())
            with urllib.request.urlopen(base+'/status') as response:
                self.assertTrue(json.load(response)['synthetic'])
            buffer = io.BytesIO(); Image.new('L', (48, 48), 80).save(buffer, format='PNG')
            with urllib.request.urlopen(urllib.request.Request(base+'/predict', data=buffer.getvalue())) as response:
                result = json.load(response)
                self.assertIn(result['label'], ['bird', 'drone'])
                self.assertTrue(result['processed'])
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(base+'/predict', data=b'not an image'))
            self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown(); server.server_close(); thread.join()

if __name__ == '__main__': unittest.main()
