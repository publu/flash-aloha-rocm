"""Convert our hand-guided YAM teaching demos into Flash ALOHA raw stream directories.

Read-only on the source. Same causal 10 Hz grid and 14-D layout as convert_h3_recordings.py.
Hand-guided (kinesthetic) takes: the action label is the measured pose (Flash's own v3 kinesthetic convention).
Images have exposure times only (image_t_s); a frame counts as available `--avail-latency` s after exposure.

usage: python -I convert_teaching_demos.py <teaching/demos> <out root> [--hz 10]
"""
import argparse, json, sys, glob
from pathlib import Path
import numpy as np
import cv2
import zarr

CAMS = ('top', 'left', 'right')
ARMS = ('left', 'right')


def latest(times, g):
    return int(np.searchsorted(times, g, side='right')) - 1


def convert(src, out_root, hz, max_joint_age, max_frame_age, avail):
    meta = json.loads((src / 'demo.json').read_text())
    jf = src / 'joints.jsonl'
    if not jf.exists():
        return None, 'no joints.jsonl'
    rows = [json.loads(l) for l in jf.read_text().splitlines() if l.strip()]
    t = np.array([r['t_s'] for r in rows], float)
    q = []
    for r in rows:
        v = []
        for a in ARMS:
            arm = r['arms'].get(a) or {}
            p = arm.get('positions_rad'); g = arm.get('gripper')
            v.extend(list(p) + [g] if p is not None and g is not None and len(p) == 6 else [np.nan] * 7)
        q.append(v)
    q = np.array(q, float)
    k = np.isfinite(q).all(1) & np.isfinite(t)
    t, q = t[k], q[k]
    o = np.argsort(t, kind='stable'); t, q = t[o], q[o]
    if len(t) < 5:
        return None, 'too few joint rows'
    cam = {}
    for n in CAMS:
        fs = [dict(k=i, acq=f['image_t_s'][n], img=f['images'][n]) for i, f in enumerate(meta.get('frames') or [])
              if f.get('image_t_s', {}).get(n) is not None and (f.get('images') or {}).get(n) and (src / f['images'][n]).exists()]
        if not fs:
            return None, f'no {n} frames'
        fs.sort(key=lambda f: f['acq'])
        cam[n] = (np.array([f['acq'] + avail for f in fs]), fs)
    start = max(t[0], *(cam[n][0][0] for n in CAMS)); end = min(t[-1], *(cam[n][0][-1] for n in CAMS))
    if end - start < 1.0:
        return None, 'overlap under 1 s'
    kept, cont, seg, prev_ok = [], [], 0, False
    for g in np.arange(start, end, 1.0 / hz):
        i = latest(t, g); ok = i >= 0 and g - t[i] <= max_joint_age
        f = {}
        if ok:
            for n in CAMS:
                j = latest(cam[n][0], g)
                if j < 0 or g - cam[n][1][j]['acq'] > max_frame_age:
                    ok = False; break
                f[n] = j
        if ok:
            if not prev_ok and kept:
                seg += 1
            kept.append((g, q[i], f)); cont.append(seg)
        prev_ok = ok
    first = {n: {} for n in CAMS}; k2 = []
    for s, c in zip(kept, cont):
        if all(first[n].setdefault(s[2][n], c) == c for n in CAMS):
            k2.append((s, c))
    kept, cont = [s for s, _ in k2], [c for _, c in k2]
    remap = {v: i for i, v in enumerate(dict.fromkeys(cont))}; cont = [remap[c] for c in cont]
    if len(kept) < 2 * hz:
        return None, f'only {len(kept)} valid ticks'
    out = out_root / f'demo-{src.name}'
    if out.exists():
        return None, 'exists'
    out.mkdir(parents=True)
    stamps = np.array([s[0] for s in kept]); qpos = np.array([s[1] for s in kept])
    qvel = np.zeros_like(qpos)
    for i in range(1, len(kept)):
        if cont[i] == cont[i - 1]:
            qvel[i] = (qpos[i] - qpos[i - 1]) * hz
    fields = dict(qpos=qpos, qvel=qvel, commanded_action=qpos.copy(), timestamps=stamps, elapsed_seconds=stamps - stamps[0],
                  continuity_id=np.array(cont, 'i8'), metadata_id=np.zeros(len(kept), 'i8'),
                  intervention_active=np.zeros(len(kept), 'u1'))
    fields['camera_observation/snapshot_timestamp'] = stamps
    for n in CAMS:
        av, fs = cam[n]
        used = sorted({s[2][n] for s in kept}); seg_of = {}
        for s, c in zip(kept, cont):
            seg_of.setdefault(s[2][n], c)
        (out / f'{n}.zarr').mkdir(); writer = None; timing = []
        for vi, fi in enumerate(used):
            img = cv2.imread(str(src / fs[fi]['img']), cv2.IMREAD_COLOR)
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(str(out / f'{n}.zarr/main.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), hz, (w, h))
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
            timing.append(dict(video_frame=vi, frame_number=int(fs[fi]['k']), capture_timestamp=float(fs[fi]['acq']),
                               continuity_id=int(seg_of[fi]), consumer_received_timestamp=float(av[fi])))
        writer.release()
        (out / f'{n}.timing.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in timing))
        fields[f'camera_observation/{n}/frame_number'] = np.array([int(fs[s[2][n]]['k']) for s in kept])
        fields[f'camera_observation/{n}/capture_timestamp'] = np.array([float(fs[s[2][n]]['acq']) for s in kept])
        fields[f'camera_observation/{n}/consumer_received_timestamp'] = np.array([float(av[s[2][n]]) for s in kept])
    control = zarr.open_group(str(out / 'control.zarr'), mode='w')
    control.attrs['action_command_style'] = 'absolute'
    for kk, v in fields.items():
        control.create_dataset(kk, data=v)
    md = dict(action_command_style='absolute', source='yam-teaching-demo', source_dir=str(src),
              objective=meta.get('objective'), arm=meta.get('arm'), status=meta.get('status'), grid_hz=hz,
              layout='[left q1..q6, left jaw, right q1..q6, right jaw]',
              action_label='measured pose (kinesthetic hand-guided take)')
    (out / 'manifest.json').write_text(json.dumps(dict(complete=True, retained=True, alignable=True,
        sample_count=len(kept), fps=hz, control_freq=hz, camera_names=list(CAMS), metadata=md), indent=1))
    (out / 'alignment_report.json').write_text(json.dumps(dict(alignable=True, errors=[])))
    return out, dict(ticks=len(kept), segments=len(remap), seconds=round(len(kept) / hz, 1), objective=meta.get('objective'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src'); p.add_argument('out'); p.add_argument('--hz', type=float, default=10.0)
    p.add_argument('--max-joint-age', type=float, default=0.25); p.add_argument('--max-frame-age', type=float, default=0.4)
    p.add_argument('--avail-latency', type=float, default=0.05)
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); report = []
    for d in sorted(Path(x).parent for x in glob.glob(f'{a.src}/*/demo.json')):
        try:
            o, info = convert(d, out, a.hz, a.max_joint_age, a.max_frame_age, a.avail_latency)
        except Exception as e:
            o, info = None, f'error: {type(e).__name__}: {e}'
        report.append(dict(src=str(d), out=str(o) if o else None, info=info)); print(json.dumps(report[-1]), flush=True)
    (out / 'conversion-report-demos.json').write_text(json.dumps(report, indent=1))
    ok = [r for r in report if r['out']]
    print(f"converted {len(ok)}/{len(report)}; {sum(r['info']['seconds'] for r in ok)/3600:.2f} h", file=sys.stderr)


if __name__ == '__main__':
    main()
