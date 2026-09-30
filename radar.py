"""Small, fully trainable NumPy CNN for grayscale radar spectrograms."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

SIZE = 48
LABELS = ['bird', 'drone']

def preprocess(image, denoise=True):
    image = image.convert('L').resize((SIZE, SIZE))
    if denoise:
        image = image.filter(ImageFilter.MedianFilter(3))
    x = np.asarray(image, dtype=np.float32) / 255
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x-lo)/max(float(hi-lo), .05), 0, 1)

def patches(x):
    return np.lib.stride_tricks.sliding_window_view(x, (5, 5), axis=(1, 2))[:, ::2, ::2]

class CNN:
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.params = dict(kernel=rng.normal(0, .15, (8, 5, 5)).astype('float32'),
                           bias=np.zeros(8, 'float32'),
                           weight=rng.normal(0, .15, (8,)).astype('float32'),
                           output=np.zeros(1, 'float32'))

    def forward(self, x):
        p = patches(x)
        z = np.einsum('nhwij,fij->nhwf', p, self.params['kernel'], optimize=True) + self.params['bias']
        features = np.maximum(z, 0).mean(axis=(1, 2))
        logits = features @ self.params['weight'] + self.params['output'][0]
        prob = 1/(1+np.exp(-np.clip(logits, -40, 40)))
        return prob, (p, z, features)

    def gradients(self, x, y):
        prob, (p, z, features) = self.forward(x)
        delta = (prob-y)/len(y)
        dz = delta[:, None, None, None]*self.params['weight'][None, None, None, :]*(z>0)/(z.shape[1]*z.shape[2])
        grads = dict(kernel=np.einsum('nhwf,nhwij->fij', dz, p, optimize=True),
                     bias=dz.sum(axis=(0, 1, 2)), weight=features.T@delta,
                     output=np.array([delta.sum()]))
        return grads

    def save(self, path, threshold, synthetic):
        np.savez_compressed(path, **self.params, threshold=threshold, synthetic=synthetic)

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(path, allow_pickle=False) as data:
            for key in model.params:
                model.params[key] = data[key].copy()
            model.threshold = float(data['threshold'])
            model.synthetic = bool(data['synthetic'])
        return model

def generate(root, count=120, seed=7):
    """Illustrative patterns only; these are not physical radar simulations."""
    if count < 10:
        raise ValueError('Use at least 10 samples per class per split.')
    rng = np.random.default_rng(seed)
    yy, tt = np.mgrid[0:SIZE, 0:SIZE]
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Output data folder must be empty to prevent stale samples.')
    for split, n in [('train', count), ('val', max(20, count//3)), ('test', max(20, count//3))]:
        for label in LABELS:
            dest = root/split/label
            dest.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                center = rng.uniform(20, 28)
                phase = rng.uniform(0, 2*np.pi)
                a = .12 + rng.normal(0, rng.uniform(.03, .1), (SIZE, SIZE))
                a += .4*np.exp(-((yy-center)/1.8)**2)
                if label == 'drone':
                    for offset in [-14, -8, 8, 14]:
                        ridge = center+offset+1.2*np.sin(tt*.8+phase)
                        a += rng.uniform(.25, .45)*np.exp(-((yy-ridge)/1.1)**2)
                else:
                    ridge = center+rng.uniform(7, 13)*np.sin(tt*rng.uniform(.15, .3)+phase)
                    a += .65*np.exp(-((yy-ridge)/2.2)**2)
                Image.fromarray((np.clip(a, 0, 1)*255).astype('uint8')).save(dest/f'{i:04}.png')
    (root/'synthetic.json').write_text(json.dumps({'seed': seed, 'description': 'Illustrative synthetic patterns; not real radar data.'}))

def dataset(root):
    sets, hashes = {}, {}
    for split in ['train', 'val', 'test']:
        images, labels = [], []
        for target, label in enumerate(LABELS):
            files = sorted(p for p in (Path(root)/split/label).glob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'})
            if not files:
                raise ValueError(f'No images in {split}/{label}')
            for path in files:
                with Image.open(path) as im:
                    x = preprocess(im)
                digest = hashlib.sha256(x.tobytes()).hexdigest()
                if digest in hashes and hashes[digest] != split:
                    raise ValueError(f'Duplicate image across splits: {path}')
                hashes[digest] = split
                images.append(x)
                labels.append(target)
        sets[split] = (np.stack(images), np.array(labels, dtype='float32'))
    return sets

def metrics(y, p, threshold):
    pred = p >= threshold
    tp = int(((y==1)&pred).sum()); tn = int(((y==0)&~pred).sum())
    fp = int(((y==0)&pred).sum()); fn = int(((y==1)&~pred).sum())
    return dict(accuracy=(tp+tn)/len(y), drone_precision=tp/max(tp+fp, 1),
                drone_recall=tp/max(tp+fn, 1), bird_false_alarm_rate=fp/max(fp+tn, 1),
                confusion_matrix=[[tn, fp], [fn, tp]], samples=len(y))

def threshold_for(y, p, min_recall=.95):
    candidates = np.unique(np.r_[0., p, 1.])
    valid = [(t, metrics(y, p, t)) for t in candidates]
    valid = [(t, m) for t, m in valid if m['drone_recall'] >= min_recall]
    return float(min(valid, key=lambda v: (v[1]['bird_false_alarm_rate'], -v[0]))[0])

def train(root, out, epochs=35, seed=7, recall=.95):
    if epochs < 1 or not 0 < recall <= 1:
        raise ValueError('epochs must be positive; recall must be in (0, 1].')
    sets = dataset(root)
    model, rng = CNN(seed), np.random.default_rng(seed)
    moment = {k: np.zeros_like(v) for k, v in model.params.items()}
    variance = {k: np.zeros_like(v) for k, v in model.params.items()}
    step, best, history = 0, float('inf'), []
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    synthetic = (Path(root)/'synthetic.json').exists()
    for epoch in range(epochs):
        x, y = sets['train']; order = rng.permutation(len(y))
        for start in range(0, len(y), 24):
            ids = order[start:start+24]
            batch = x[ids].copy()
            # Time shifts and additive receiver noise; preserve Doppler orientation.
            for j in range(len(batch)):
                batch[j] = np.roll(batch[j], int(rng.integers(-3, 4)), axis=1)
            batch = np.clip(batch+rng.normal(0, .025, batch.shape), 0, 1).astype('float32')
            grads = model.gradients(batch, y[ids]); step += 1
            for k, g in grads.items():
                moment[k] = .9*moment[k]+.1*g
                variance[k] = .999*variance[k]+.001*g*g
                model.params[k] -= .01*(moment[k]/(1-.9**step))/(np.sqrt(variance[k]/(1-.999**step))+1e-8)
        vx, vy = sets['val']; vp = model.forward(vx)[0]
        loss = float(-np.mean(vy*np.log(vp+1e-7)+(1-vy)*np.log(1-vp+1e-7)))
        history.append(dict(epoch=epoch+1, validation_loss=loss, **metrics(vy, vp, .5)))
        if loss < best:
            best = loss
            model.save(out/'model.npz', .5, synthetic)
        print(f'Epoch {epoch+1:02}: validation loss {loss:.4f}', flush=True)
    model = CNN.load(out/'model.npz')
    vp = model.forward(sets['val'][0])[0]
    threshold = threshold_for(sets['val'][1], vp, recall)
    model.save(out/'model.npz', threshold, synthetic)
    tp = model.forward(sets['test'][0])[0]
    report = dict(data_type='synthetic demonstration' if synthetic else 'user supplied radar images',
                  seed=seed, threshold=threshold, validation_recall_target=recall,
                  test=metrics(sets['test'][1], tp, threshold),
                  test_default_threshold=metrics(sets['test'][1], tp, .5), history=history,
                  note='Synthetic results do not establish real-world accuracy. Rows: bird/drone; columns: bird/drone.')
    (out/'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['test'], indent=2))

def predict(model, image):
    p = float(model.forward(preprocess(image)[None])[0][0])
    return dict(label='drone' if p >= model.threshold else 'bird', drone_score=p,
                threshold=model.threshold, synthetic_model=model.synthetic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate'); g.add_argument('--data', default='data'); g.add_argument('--count', type=int, default=120)
    t = sub.add_parser('train'); t.add_argument('--data', default='data'); t.add_argument('--out', default='artifacts'); t.add_argument('--epochs', type=int, default=35); t.add_argument('--recall', type=float, default=.95)
    p = sub.add_parser('predict'); p.add_argument('image'); p.add_argument('--model', default='artifacts/model.npz')
    args = parser.parse_args()
    if args.command == 'generate': generate(args.data, args.count)
    elif args.command == 'train': train(args.data, args.out, args.epochs, recall=args.recall)
    else:
        with Image.open(args.image) as im: print(json.dumps(predict(CNN.load(args.model), im), indent=2))
