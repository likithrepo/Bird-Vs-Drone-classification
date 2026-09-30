"""Local browser interface. Run: python app.py"""
import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from radar import CNN, predict, preprocess

PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RadarScope · Drone & Bird Classifier</title><style>
*{box-sizing:border-box}body{margin:0;background:#091321;color:#e7edf6;font:16px system-ui}main{max-width:1000px;margin:70px auto;padding:24px}.eyebrow{color:#4ce1be;font-size:12px;letter-spacing:3px}h1{font-size:46px;margin:20px 0}p{line-height:1.6;color:#a8b6cb}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.card{background:#111f31;border:1px solid #26364d;border-radius:18px;padding:28px}input{width:100%;margin:24px 0;color:#bccce2}button{background:#4ce1be;border:0;border-radius:9px;padding:14px 22px;font-weight:700;cursor:pointer}button:disabled{opacity:.4}img{width:100%;image-rendering:pixelated;margin-top:20px;border-radius:10px}#label{font-size:38px;text-transform:capitalize;color:#4ce1be}progress{width:100%;accent-color:#4ce1be}small{color:#a8b6cb}#error{color:#ffb6a5}.badge{display:inline-block;border:1px solid #bd9352;color:#ffd091;border-radius:20px;padding:7px 14px;font-size:13px}details{margin-top:24px}pre{overflow:auto;white-space:pre-wrap;font-size:12px}@media(max-width:700px){.grid{grid-template-columns:1fr}h1{font-size:32px}main{margin:20px auto}}
</style><main><div class="eyebrow">RADARSCOPE / SIGNAL INTELLIGENCE</div><h1>Drone or bird?<br>Read the radar signature.</h1><p>A trainable convolutional network for grayscale radar imagery and micro-Doppler spectrograms.</p><div class="badge" id="mode">Loading model status…</div><div class="grid" style="margin-top:30px"><section class="card"><h2>Analyze a signal</h2><p>Upload a radar spectrogram. Noise filtering and intensity normalization run before classification.</p><label for="file">Radar image</label><input id="file" type="file" accept="image/png,image/jpeg,image/bmp,image/tiff"><button id="run">Classify image</button><p id="error" role="alert"></p><img id="preview" alt="Uploaded radar image" hidden></section><section class="card" aria-live="polite"><h2>Classification</h2><div id="label">Awaiting signal</div><p id="score">Upload an image to begin.</p><progress id="prob" max="1" value="0" aria-label="Drone model score"></progress><p id="threshold"></p><small>The score is a model output, not a calibrated probability. A binary classifier cannot reliably reject unknown targets.</small><img id="processed" alt="Preprocessed grayscale radar image" hidden></section></div><details><summary>Held-out evaluation report</summary><pre id="metrics"></pre></details><p>Research prototype · Real radar recordings are required to validate operational performance.</p></main><script>
const $=id=>document.getElementById(id);let previewUrl;
fetch('/status').then(r=>r.json()).then(s=>{$('mode').textContent=s.synthetic?'SYNTHETIC DEMO MODEL · not validated on real radar':'USER-TRAINED MODEL';$('metrics').textContent=JSON.stringify(s.metrics,null,2)}).catch(()=>{$('mode').textContent='Status unavailable'});
$('file').onchange=()=>{if(previewUrl)URL.revokeObjectURL(previewUrl);const f=$('file').files[0];if(f){previewUrl=URL.createObjectURL(f);$('preview').src=previewUrl;$('preview').hidden=false}};
$('run').onclick=async()=>{const f=$('file').files[0];if(!f){$('error').textContent='Choose an image first.';return}if(f.size>10*1024*1024){$('error').textContent='Use an image under 10 MB.';return}$('run').disabled=true;$('error').textContent='';try{const r=await fetch('/predict',{method:'POST',body:f});const s=await r.json();if(!r.ok)throw Error(s.error);$('label').textContent=s.label;$('score').textContent='Drone score: '+(s.drone_score*100).toFixed(1)+'%';$('prob').value=s.drone_score;$('threshold').textContent='Drone decision threshold: '+s.threshold.toFixed(3);$('processed').src='data:image/png;base64,'+s.processed;$('processed').hidden=false}catch(e){$('error').textContent=e.message}finally{$('run').disabled=false}};
</script></html>'''

def handler(model, metrics):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload, content_type='application/json'):
            data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
            self.send_response(status); self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            if self.path == '/': self.reply(200, PAGE, 'text/html; charset=utf-8')
            elif self.path == '/status': self.reply(200, dict(synthetic=model.synthetic, metrics=metrics))
            else: self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path != '/predict': return self.reply(404, {'error': 'Not found'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 10*1024*1024: return self.reply(413, {'error': 'Use an image under 10 MB.'})
                with Image.open(io.BytesIO(self.rfile.read(length))) as im:
                    if im.width*im.height > 20_000_000: return self.reply(413, {'error': 'Image dimensions exceed 20 megapixels.'})
                    result = predict(model, im)
                    processed = Image.fromarray((preprocess(im)*255).astype('uint8'))
                    buffer = io.BytesIO(); processed.save(buffer, format='PNG')
                    result['processed'] = base64.b64encode(buffer.getvalue()).decode()
                self.reply(200, result)
            except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
                self.reply(400, {'error': 'Upload a valid supported image.'})
    return Handler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='artifacts/model.npz'); parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    path = Path(args.model)
    if not path.exists(): parser.error('Model missing. Run radar.py generate and radar.py train first.')
    report = path.parent/'metrics.json'
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(CNN.load(path), json.loads(report.read_text()) if report.exists() else {}))
    print(f'Open http://127.0.0.1:{args.port}', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: server.server_close()
