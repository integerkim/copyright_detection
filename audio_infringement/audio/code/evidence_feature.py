"""In-house acoustic evidence used by the deployment scoring pipeline."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: Path, sample_rate: int = 22050) -> tuple[np.ndarray,int]:
    audio,sr=sf.read(path,always_2d=False,dtype="float32")
    if audio.ndim==2: audio=audio.mean(axis=1)
    if sr!=sample_rate: audio=librosa.resample(audio,orig_sr=sr,target_sr=sample_rate); sr=sample_rate
    return np.asarray(audio,dtype=np.float32),sr


def _subsequence_cost(reference: np.ndarray,query: np.ndarray) -> float:
    reference=np.nan_to_num(reference,nan=0.0)+1e-8; query=np.nan_to_num(query,nan=0.0)+1e-8
    distance,_=librosa.sequence.dtw(X=reference,Y=query,metric="cosine",subseq=True,backtrack=True)
    endpoint=int(np.argmin(distance[-1])); return float(distance[-1,endpoint]/max(1,reference.shape[1]))


def acoustic_sequence_similarity(reference_high: dict,query_high: dict) -> dict:
    """Reuse the High-level MFCC trajectories instead of extracting MFCC twice."""
    r_mfcc=reference_high["timbre_trajectory"]
    q_mfcc=query_high["timbre_trajectory"]
    forward=_subsequence_cost(r_mfcc,q_mfcc); reverse=_subsequence_cost(r_mfcc,q_mfcc[:,::-1])
    similarity=float(np.exp(-4*forward)); directionality=float(reverse/(forward+reverse+1e-8))
    return {"score":similarity,"mfcc_cost":forward,"reverse_mfcc_cost":reverse,"directionality":directionality}


def fingerprint_match(reference: np.ndarray,query: np.ndarray) -> dict:
    """Match spectral frame signatures only when they share one time offset."""
    def tokens(audio):
        spectrum=np.abs(librosa.stft(audio,n_fft=2048,hop_length=512)); result=[]
        for time in range(spectrum.shape[1]):
            peaks=np.argpartition(spectrum[:,time],-min(5,spectrum.shape[0]))[-5:]
            token=hashlib.sha1(",".join(map(str,sorted(peaks.tolist()))).encode()).hexdigest()[:12]
            result.append((token,time))
        return result
    ref=tokens(reference); qry=tokens(query); index=defaultdict(list)
    for token,time in ref: index[token].append(time)
    offsets=Counter(ref_time-qry_time for token,qry_time in qry for ref_time in index.get(token,[]))
    offset,count=offsets.most_common(1)[0] if offsets else (0,0)
    score=count/max(1,min(len(ref),len(qry)))
    return {"score":float(min(1,score)),"dominant_offset_frames":int(offset),"aligned_hashes":int(count),
            "reference_frames":len(ref),"query_frames":len(qry)}


def extract_low_level(audio: np.ndarray,sr: int) -> dict:
    """Frame/event observations. These are measurements, not similarity scores."""
    hop=512
    onset=librosa.onset.onset_strength(y=audio,sr=sr,hop_length=hop)
    onset_frames=librosa.onset.onset_detect(onset_envelope=onset,sr=sr,hop_length=hop)
    tempo,beat_frames=librosa.beat.beat_track(onset_envelope=onset,sr=sr,hop_length=hop,units="frames")
    return {
        "onset_envelope":onset.astype(np.float32),
        "onset_times":librosa.frames_to_time(onset_frames,sr=sr,hop_length=hop).astype(np.float32),
        "bpm":np.asarray([float(np.asarray(tempo).flat[0])],dtype=np.float32),
        "beat_frames":np.asarray(beat_frames,dtype=np.int32),
        "beat_times":librosa.frames_to_time(beat_frames,sr=sr,hop_length=hop).astype(np.float32),
        "mfcc_frames":librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=20,hop_length=hop).astype(np.float32),
    }


def build_high_level(low: dict) -> dict:
    """Convert observations into ordered musical patterns."""
    onset=low["onset_envelope"]; beats=low["beat_frames"]; groove=[]
    for left,right in zip(beats[:-1],beats[1:]):
        source=np.arange(left,right+1); target=np.linspace(left,right,16)
        row=np.interp(target,source,onset[left:right+1]); row/=np.linalg.norm(row)+1e-8; groove.append(row)
    groove_sequence=np.asarray(groove,dtype=np.float32).T if groove else np.zeros((16,1),dtype=np.float32)
    return {
        "groove_sequence":groove_sequence,
        "timbre_trajectory":low["mfcc_frames"][1:].astype(np.float32),
    }


def rhythm_similarity(reference_high: dict,query_high: dict) -> dict:
    cost=_subsequence_cost(reference_high["groove_sequence"],query_high["groove_sequence"])
    return {"score":float(np.exp(-4*cost)),"cost":cost,
            "reference_beats":int(reference_high["groove_sequence"].shape[1]),
            "query_beats":int(query_high["groove_sequence"].shape[1])}


def save_feature_levels(low: dict,high: dict,base: Path) -> dict:
    base.parent.mkdir(parents=True,exist_ok=True)
    low_path=base.with_name(base.name+"_low.npz"); high_path=base.with_name(base.name+"_high.npz")
    np.savez_compressed(low_path,**low); np.savez_compressed(high_path,**high)
    return {"low_level_file":str(low_path),"high_level_file":str(high_path)}
