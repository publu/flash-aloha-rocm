"""Convert our YAM robot execution recordings (recording.json + joints.npz + camera frames) into Flash ALOHA raw stream directories.

Read-only on the source. Causal 10 Hz grid: at each tick g every value is the latest one *known at g*
(measured joints, commanded target, newest camera frame received). No interpolation, no future data.
qpos / commanded_action layout (14): [left q1..q6, left jaw, right q1..q6, right jaw].
An arm with no command in force at g (not streamed) is labelled with its own measured pose (station holds it).
Ticks where any camera's newest frame or the joint sample is staler than the limits split continuity.

usage: python -I convert_h3_recordings.py <video-motion root> <out root> [--hz 10] [--limit N]
"""
import argparse, json, os, sys, glob
from pathlib import Path
import numpy as np
import cv2
import zarr

CAMS = ('top', 'left', 'right')
ARMS = ('left', 'right')


def load(rec):
    m = json.loads((rec / 'recording.json').read_text())
    z = np.load(rec / 'joints.npz', allow_pickle=False)
    cols = [str(c) for c in z['columns']]
    rows = z['rows']
    t = rows[:, cols.index('t')]
    t = np.where(np.isfinite(t), t, rows[:, cols.index('t_tap')])
    frames = [json.loads(l) for l in (rec / 'frames.jsonl').read_text().splitlines() if l.strip()]
    return m, cols, rows, t, frames


def latest(times, g):
    """Index of the last time <= g, or -1."""
    return int(np.searchsorted(times, g, side='right')) - 1


def convert(rec, out_root, hz, max_joint_age, max_frame_age):
    m, cols, rows, t, frames = load(rec)
    ok = np.isfinite(t)
    rows, t = rows[ok], t[ok]
    order = np.argsort(t, kind='stable'); rows, t = rows[order], t[order]
    if len(t) < 5:
        return None, 'too few joint rows'
    c = {n: i for i, n in enumerate(cols)}
    # per-arm measured / commanded series, each with its own valid timestamps (forward-filled only by "latest <= g")
    meas, cmd = {}, {}
    for a in ARMS:
        qi = [c[f'{a}_q{j}'] for j in range(1, 7)] + [c[f'{a}_jaw']]
        ci = [c[f'{a}_c{j}'] for j in range(1, 7)] + [c[f'{a}_jaw_cmd']]
        q = rows[:, qi]; k = np.isfinite(q).all(1); meas[a] = (t[k], q[k])
        u = rows[:, ci]; k = np.isfinite(u).all(1); cmd[a] = (t[k], u[k])
    # camera frames: receive time t is when it became available to a consumer; acq is the exposure
    cam = {}
    for name in CAMS:
        fs = sorted((f for f in frames if f['cam'] == name and f.get('jpg')), key=lambda f: f['t'])
        if not fs:
            return None, f'no {name} frames'
        cam[name] = (np.array([f['t'] for f in fs]), fs)
    start = max(meas['left'][0][0] if len(meas['left'][0]) else np.inf,
                meas['right'][0][0] if len(meas['right'][0]) else np.inf,
                *(cam[n][0][0] for n in CAMS))
    end = min(t[-1], *(cam[n][0][-1] for n in CAMS))
    if not np.isfinite(start) or end - start < 1.0:
        return None, 'overlap under 1 s'
    grid = np.arange(start, end, 1.0 / hz)
    samples = []
    for g in grid:
        s = dict(t=g, q=[], u=[], f={}, ok=True)
        for a in ARMS:
            mt, mq = meas[a]; i = latest(mt, g)
            if i < 0 or g - mt[i] > max_joint_age:
                s['ok'] = False; break
            s['q'].append(mq[i])
            ct, cu = cmd[a]; j = latest(ct, g)
            # a command is "in force" only if it is newer than the measured sample's lease gap limit
            s['u'].append(cu[j] if j >= 0 and g - ct[j] <= max_joint_age else mq[i])
        if s['ok']:
            for n in CAMS:
                ft, fs = cam[n]; i = latest(ft, g)
                if i < 0 or g - fs[i]['acq'] > max_frame_age:
                    s['ok'] = False; break
                s['f'][n] = i
        samples.append(s)
    # continuity: a new segment after every invalid tick; drop invalid ticks
    seg, cont, kept = 0, [], []
    prev_ok = False
    for s in samples:
        if s['ok']:
            if not prev_ok and kept:
                seg += 1
            kept.append(s); cont.append(seg)
        prev_ok = s['ok']
    if len(kept) < 2 * hz:
        return None, f'only {len(kept)} valid ticks'
    # a frame must belong to one continuity segment (Flash sidecar contract): drop ticks reusing a frame across a break
    first_seg = {n: {} for n in CAMS}
    keep2 = []
    for s, k in zip(kept, cont):
        good = all(first_seg[n].setdefault(s['f'][n], k) == k for n in CAMS)
        if good:
            keep2.append((s, k))
    kept, cont = [s for s, _ in keep2], [k for _, k in keep2]
    # renumber continuity densely
    remap = {v: i for i, v in enumerate(dict.fromkeys(cont))}; cont = [remap[k] for k in cont]
    out = out_root / f"{rec.parent.name}-{rec.name}"
    if out.exists():
        return None, 'exists'
    out.mkdir(parents=True)
    t0 = kept[0]['t']
    stamps = np.array([s['t'] for s in kept])
    qpos = np.array([np.concatenate(s['q']) for s in kept])
    act = np.array([np.concatenate(s['u']) for s in kept])
    qvel = np.zeros_like(qpos)
    for i in range(1, len(kept)):
        if cont[i] == cont[i - 1]:
            qvel[i] = (qpos[i] - qpos[i - 1]) * hz
    fields = dict(qpos=qpos, qvel=qvel, commanded_action=act, timestamps=stamps, elapsed_seconds=stamps - t0,
                  continuity_id=np.array(cont, dtype='i8'), metadata_id=np.zeros(len(kept), 'i8'),
                  intervention_active=np.zeros(len(kept), 'u1'))
    fields['camera_observation/snapshot_timestamp'] = stamps
    for n in CAMS:
        ft, fs = cam[n]
        used = sorted({s['f'][n] for s in kept})
        seg_of = {}
        for s, k in zip(kept, cont):
            seg_of.setdefault(s['f'][n], k)
        blob = (rec / f'{n}.mjpg').read_bytes()
        (out / f'{n}.zarr').mkdir()
        w = h = None; writer = None; timing = []
        for vi, fi in enumerate(used):
            f = fs[fi]; off, ln = f['jpg']
            img = cv2.imdecode(np.frombuffer(blob, np.uint8, count=ln, offset=off), cv2.IMREAD_COLOR)
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(str(out / f'{n}.zarr/main.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), hz, (w, h))
                if not writer.isOpened():
                    raise RuntimeError('video writer failed')
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
            timing.append(dict(video_frame=vi, frame_number=int(f['fid']), capture_timestamp=float(f['acq']),
                               continuity_id=int(seg_of[fi]), consumer_received_timestamp=float(f['t'])))
        writer.release()
        (out / f'{n}.timing.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in timing))
        fields[f'camera_observation/{n}/frame_number'] = np.array([int(fs[s['f'][n]]['fid']) for s in kept])
        fields[f'camera_observation/{n}/capture_timestamp'] = np.array([float(fs[s['f'][n]]['acq']) for s in kept])
        fields[f'camera_observation/{n}/consumer_received_timestamp'] = np.array([float(fs[s['f'][n]]['t']) for s in kept])
    control = zarr.open_group(str(out / 'control.zarr'), mode='w')
    control.attrs['action_command_style'] = 'absolute'
    for k, v in fields.items():
        control.create_dataset(k, data=v)
    meta = dict(action_command_style='absolute', source='yam-recording', source_dir=str(rec),
                task_id=m.get('task_id'), clip_id=m.get('clip_id'), label=m.get('label'), prompt=m.get('prompt'),
                grid_hz=hz, layout='[left q1..q6, left jaw, right q1..q6, right jaw]',
                action_label='latest commanded target in force at tick; idle arm = its measured pose',
                outcome='unreviewed execution (not a verified-success demonstration)')
    (out / 'manifest.json').write_text(json.dumps(dict(complete=True, retained=True, alignable=True,
        sample_count=len(kept), fps=hz, control_freq=hz, camera_names=list(CAMS), metadata=meta), indent=1))
    (out / 'alignment_report.json').write_text(json.dumps(dict(alignable=True, errors=[])))
    return out, dict(ticks=len(kept), segments=len(remap), seconds=round(len(kept) / hz, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src'); p.add_argument('out'); p.add_argument('--hz', type=float, default=10.0)
    p.add_argument('--max-joint-age', type=float, default=0.25); p.add_argument('--max-frame-age', type=float, default=0.3)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs = sorted(Path(x).parent for x in glob.glob(f'{a.src}/*/recording-*/recording.json'))
    if a.limit:
        recs = recs[:a.limit]
    report = []
    for r in recs:
        try:
            d, info = convert(r, out, a.hz, a.max_joint_age, a.max_frame_age)
        except Exception as e:
            d, info = None, f'error: {type(e).__name__}: {e}'
        report.append(dict(src=str(r), out=str(d) if d else None, info=info))
        print(json.dumps(report[-1]), flush=True)
    (out / 'conversion-report.json').write_text(json.dumps(report, indent=1))
    ok = [r for r in report if r['out']]
    print(f"converted {len(ok)}/{len(report)}; {sum(r['info']['seconds'] for r in ok)/3600:.2f} h", file=sys.stderr)


if __name__ == '__main__':
    main()
