import importlib.util
import tempfile
import unittest
from pathlib import Path

from xai_experiment.image_download import _candidates, _is_jpeg, fetch_review_image


class ImageDownloadTests(unittest.TestCase):
    def test_coco_uses_verified_google_storage_url_first(self):
        source = 'https://images.cocodataset.org/val2017/000000322829.jpg'
        self.assertEqual(_candidates(source)[0],
                         'https://storage.googleapis.com/images.cocodataset.org/val2017/000000322829.jpg')

    def test_unknown_host_is_rejected(self):
        with self.assertRaises(ValueError):
            _candidates('https://example.com/untrusted.jpg')

    def test_html_cannot_be_embedded_as_jpeg(self):
        self.assertFalse(_is_jpeg(b'<html>not an image</html>'))

    @unittest.skipUnless(importlib.util.find_spec('requests') and importlib.util.find_spec('certifi'),
                         'optional image download dependencies not installed')
    def test_certificate_failure_uses_next_verified_url_and_checks_jpeg(self):
        import requests

        source = 'https://images.cocodataset.org/val2017/000000322829.jpg'
        bucket = _candidates(source)[0]
        image = b'\xff\xd8' + b'0' * 120 + b'\xff\xd9'

        class Response:
            url = bucket
            content = image

            def raise_for_status(self):
                return None

        class Session:
            calls = []

            def get(self, url, **kwargs):
                self.calls.append((url, kwargs['verify']))
                if len(self.calls) == 1:
                    raise requests.exceptions.SSLError('certificate mismatch')
                return Response()

        # Force a retry: the first bucket request fails, then the source URL
        # would still need a trusted certificate. The fake response records the
        # actual URL to check the saved provenance.
        session = Session()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'sample.jpg'
            result = fetch_review_image(source, output, session=session)
            self.assertEqual(output.read_bytes(), image)
            self.assertTrue(result['tls_verified'])
            self.assertEqual(session.calls[0][0], bucket)
            self.assertEqual(session.calls[1][0], source)
            self.assertTrue(all(verify is not False for _, verify in session.calls))

    @unittest.skipUnless(importlib.util.find_spec('requests') and importlib.util.find_spec('certifi'),
                         'optional image download dependencies not installed')
    def test_last_resort_is_recorded_as_unverified(self):
        import requests

        source = 'https://cs.stanford.edu/people/rak248/VG_100K/example.jpg'
        image = b'\xff\xd8' + b'1' * 120 + b'\xff\xd9'

        class Response:
            url = source
            content = image

            def raise_for_status(self):
                return None

        class Session:
            def get(self, url, **kwargs):
                if kwargs['verify'] is not False:
                    raise requests.exceptions.SSLError('certificate mismatch')
                return Response()

        with tempfile.TemporaryDirectory() as directory:
            result = fetch_review_image(source, Path(directory) / 'sample.jpg',
                                        allow_unverified_tls=True, session=Session())
            self.assertFalse(result['tls_verified'])
            self.assertEqual(result['download_url'], source)
