from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs
import json
from .prediction import get_prediction # Correct relative import
import cgi

# This is the standard Vercel handler for Python serverless functions.
# It inherits from BaseHTTPRequestHandler to process HTTP requests.
class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        # Parse the form data to get the uploaded file
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': self.headers['Content-Type'],
                     })

        # Check if the 'file' field is in the form
        if 'file' in form:
            file_item = form['file']
            if file_item.file:
                # Read the image bytes from the uploaded file
                image_bytes = file_item.file.read()

                try:
                    # Get the prediction from our model
                    result = get_prediction(image_bytes)
                    
                    # Send a successful (200) response
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode('utf-8'))
                except Exception as e:
                    # If prediction fails, send a server error (500)
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
                return

        # If no file is found, send a bad request error (400)
        self.send_response(400)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": "No file uploaded."}).encode('utf-8'))
        return