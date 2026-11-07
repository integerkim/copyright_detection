"""
motion_main.py
--------------
Motion-modality similarity search against the DB.

Input : query video  OR  its preprocessed SMPL npz
Output: result.txt - the most similar DB motion (song / part / performer)
        and its similarity score (overall + PSC / AC / DC, 0-100), plus top-k ranking

Steps
  1) (if --video) run ../preprocess/motion_preprocess.py -> query SMPL npz
  2) compare query with every DB motion (motion_metric.compute_similarity)
  3) rank DB motions by similarity; report the most similar one

Usage
  python motion_main.py --video ../../../suspect_data/a.mp4 --result_dir ../../../results/a/motion
  python motion_main.py --query_npz ../../../results/a/motion/a_smpl.npz
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import motion_metric as M  # noqa: E402
import motion_canonical as C  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_RESULT = os.path.join(ROOT, "..", "results", "motion_standalone")
PREPROCESS = os.path.join(ROOT, "motion", "preprocess", "motion_preprocess.py")
PROJECT = os.path.abspath(os.path.join(ROOT, ".."))
DEFAULT_HYBRIK = os.path.join(PROJECT, "third_party", "HybrIK")
DEFAULT_DB = os.path.join(PROJECT, "copyright_database", "motion")


def describe(db_file):
    """DB file name -> (song, part, performer). e.g. cupid_chorus_1_KHJ_Skeleton 002-... -> cupid, chorus_1, KHJ"""
    stem = os.path.basename(db_file).split("-smpl_pose_traj")[0].replace(".npz", "")
    tok = stem.split("_")
    if len(tok) >= 4 and tok[3].isupper():
        return tok[0], f"{tok[1]}_{tok[2]}", tok[3]
    return stem, "-", "-"


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", help="generated / suspected video")
    g.add_argument("--query_npz", help="already-extracted SMPL npz")
    p.add_argument("--db_dir", nargs="+", default=[DEFAULT_DB], help="DB 폴더 (여러 개 가능, 하위 폴더 포함)")
    p.add_argument("--result_dir", default=DEFAULT_RESULT)
    p.add_argument("--hybrik_root", default=DEFAULT_HYBRIK)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--stride_sec", type=float, default=0.5, help="window stride for subsequence search")
    p.add_argument("--no_canonical", action="store_true", help="disable swing-only pose canonicalization")
    p.add_argument("--no_subsequence", action="store_true", help="compare whole sequences only")
    p.add_argument("--topk", type=int, default=5)
    return p.parse_args()


FPS = 30
MAX_WINDOWS = 40  # cap on candidate windows per DB motion (long DB clips get a larger stride)


def _load(path, canonical):
    d = np.load(path, allow_pickle=True)
    poses = d["poses_axis_angle"].astype(np.float32)
    trans = d["trans"].astype(np.float32)
    if canonical:
        poses = C.canonicalize(poses)
    return poses, trans


def _windows(n_long, n_short, stride):
    """Start indices of length-n_short windows inside a length-n_long sequence."""
    if stride is None or n_long <= int(n_short * 1.2):  # whole-sequence comparison
        return [None]
    last = n_long - n_short
    stride = max(stride, int(np.ceil(last / (MAX_WINDOWS - 1))))
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _score_one(args):
    """
    Best similarity between the query and one DB motion.
    Subsequence search: the shorter sequence is slid over the longer one (a plagiarized clip
    usually copies only part of a DB choreography); each window is scored with the unchanged
    metric (compute_once), both for the query and its mirrored version. Best window wins.
    """
    db_path, query_path, canonical, stride = args
    try:
        P_db, _ = _load(db_path, canonical)
        P_q, T_q = _load(query_path, canonical)
        P_qm = M.mirror_motion(P_q, T_q)[0]
        db_long = len(P_db) >= len(P_q)
        n_long, n_short = (len(P_db), len(P_q)) if db_long else (len(P_q), len(P_db))
        best = None
        for s in _windows(n_long, n_short, stride):
            for mirror, Pq_ in ((False, P_q), (True, P_qm)):
                if s is None:
                    ref, qry = P_db, Pq_
                elif db_long:
                    ref, qry = P_db[s:s + n_short], Pq_
                else:
                    ref, qry = P_db, Pq_[s:s + n_short]
                r = M.compute_once(ref, qry)
                if best is None or r["S_FINAL"] > best["S_FINAL"]:
                    best = dict(db=db_path, S_PSC=r["S_PSC"], S_AC=r["S_AC"], S_DC=r["S_DC"],
                                S_FINAL=r["S_FINAL"], mirror=mirror,
                                seg=(0, len(P_db)) if s is None or not db_long else (s, s + n_short))
        return best
    except Exception as e:  # keep going if one DB file is broken
        return dict(db=db_path, error=str(e))


def to100(x):
    return float(np.clip(x, 0.0, 1.0) * 100.0)


def main():
    a = parse_args()
    os.makedirs(a.result_dir, exist_ok=True)

    # 1) preprocess
    if a.video:
        name = os.path.splitext(os.path.basename(a.video))[0]
        query = os.path.join(a.result_dir, f"{name}_smpl.npz")
        if not os.path.exists(query):
            subprocess.run([sys.executable, PREPROCESS, "--video", a.video, "--out", query,
                            "--hybrik_root", a.hybrik_root, "--gpu", str(a.gpu)], check=True)
        src = a.video
    else:
        query, src = a.query_npz, a.query_npz
        name = os.path.splitext(os.path.basename(query))[0]

    q_len = np.load(query)["poses_axis_angle"].shape[0]

    # 2) compare with DB
    db_files = sorted({f for d in a.db_dir for f in glob.glob(os.path.join(d, "**", "*.npz"), recursive=True)})
    assert db_files, f"No DB npz in {a.db_dir}"
    t0 = time.time()
    with Pool(a.workers) as pool:
        stride = None if a.no_subsequence else max(1, int(a.stride_sec * FPS))
        rows = pool.map(_score_one, [(d, query, not a.no_canonical, stride) for d in db_files])
    ok = [r for r in rows if "error" not in r]
    ok.sort(key=lambda r: r["S_FINAL"], reverse=True)
    best = ok[0]

    # 3) scores (0-100)
    song, part, performer = describe(best["db"])

    lines = [
        "==== Motion Similarity Search ====",
        f"Input              : {os.path.basename(src)}",
        f"Query frames       : {q_len} (30 fps)",
        f"DB size            : {len(ok)} motions",
        "",
        "---- Most similar DB motion ----",
        f"DB file            : {os.path.basename(best['db'])}",
        f"Song / Part        : {song} / {part}",
        f"Performer          : {performer}",
        f"Mirrored match     : {'yes' if best['mirror'] else 'no'}",
        f"Matched DB segment : {best['seg'][0] / FPS:.1f}s - {best['seg'][1] / FPS:.1f}s",
        "",
        "---- Similarity score (0-100) ----",
        f"Overall            : {to100(best['S_FINAL']):6.2f}",
        f"  PSC (Pose Signature Consistency) : {to100(best['S_PSC']):6.2f}",
        f"  AC  (Articulation Consistency)   : {to100(best['S_AC']):6.2f}",
        f"  DC  (Dynamic Flow Consistency)   : {to100(best['S_DC']):6.2f}",
        "  Overall = 0.5*PSC + 0.3*AC + 0.2*DC",
        "",
        f"---- Top-{a.topk} most similar DB motions ----",
        f"{'rank':>4} {'overall':>7} {'PSC':>6} {'AC':>6} {'DC':>6}  {'song':<14} {'part':<10} {'performer':<9} file",
    ]
    for i, r in enumerate(ok[: a.topk], 1):
        s_, p_, f_ = describe(r["db"])
        lines.append(f"{i:>4} {to100(r['S_FINAL']):7.2f} {to100(r['S_PSC']):6.2f} {to100(r['S_AC']):6.2f} "
                     f"{to100(r['S_DC']):6.2f}  {s_:<14} {p_:<10} {f_:<9} {os.path.basename(r['db'])}")
    if len(ok) < len(rows):
        lines.append(f"\n[warn] {len(rows) - len(ok)} DB files failed to load")
    lines.append(f"\n(elapsed {time.time() - t0:.1f}s)")

    text = "\n".join(lines)
    print(text)
    out_txt = os.path.join(a.result_dir, "result.txt")
    with open(out_txt, "w") as f:
        f.write(text + "\n")

    # full ranking for reference
    with open(os.path.join(a.result_dir, "all_db_scores.csv"), "w") as f:
        f.write("rank,db_file,song,part,performer,overall,PSC,AC,DC,mirror\n")
        for i, r in enumerate(ok, 1):
            s_, p_, f_ = describe(r["db"])
            f.write(f"{i},\"{os.path.basename(r['db'])}\",{s_},{p_},{f_},{to100(r['S_FINAL']):.4f},"
                    f"{to100(r['S_PSC']):.4f},{to100(r['S_AC']):.4f},{to100(r['S_DC']):.4f},{r['mirror']}\n")
    # machine-readable summary (used by the integrated pipeline)
    import json
    def _row(r):
        s_, p_, f_ = describe(r["db"])
        return dict(db_file=os.path.basename(r["db"]), song=s_, part=p_, performer=f_,
                    overall=to100(r["S_FINAL"]), PSC=to100(r["S_PSC"]), AC=to100(r["S_AC"]),
                    DC=to100(r["S_DC"]), mirror=bool(r["mirror"]),
                    segment_sec=[r["seg"][0] / FPS, r["seg"][1] / FPS])
    with open(os.path.join(a.result_dir, "result.json"), "w") as f:
        json.dump(dict(input=os.path.basename(src), query_npz=query, query_frames=int(q_len),
                       db_size=len(ok), best=_row(best), top=[_row(r) for r in ok[: a.topk]]),
                  f, indent=1, ensure_ascii=False)
    print(f"\nsaved: {out_txt}")


if __name__ == "__main__":
    main()
