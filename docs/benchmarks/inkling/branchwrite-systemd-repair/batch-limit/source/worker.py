import json,os,secrets,socketserver,sys
from pathlib import Path
config=json.loads(Path(__file__).with_name('service.json').read_text())
if config['mode'] != 'serve':
    raise ValueError('report-worker requires mode serve')
if type(config['batch_limit']) is not int or not 1 <= config['batch_limit'] <= 256:
    raise ValueError('batch_limit must be a positive JSON integer up to 256')
if '--check' not in sys.argv:
    class Health(socketserver.StreamRequestHandler):
        def handle(self):
            greeting = 'ready-' + secrets.token_hex(8)
            print(greeting,flush=True)
            self.wfile.write((greeting+'\n').encode())
    with socketserver.UnixStreamServer(os.environ['HEALTH_ENDPOINT'],Health) as service:
        service.serve_forever()
