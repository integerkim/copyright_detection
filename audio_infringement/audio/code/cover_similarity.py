"""Research-backed tonal version identification using Essentia.

Pipeline: HPCP -> OTI binary chroma cross-similarity -> constrained
Smith-Waterman local alignment (chen17). Lower distance means more similar.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def _hpcp(path: Path) -> np.ndarray:
    import essentia.standard as es
    from essentia.pytools.spectral import hpcpgram
    sample_rate=32000
    audio=es.MonoLoader(filename=str(path),sampleRate=sample_rate)()
    return hpcpgram(audio,sampleRate=sample_rate,frameSize=2048,hopSize=1024)


def cover_song_evidence(reference: Path, query: Path) -> dict:
    import essentia.standard as es
    reference_hpcp=_hpcp(reference); query_hpcp=_hpcp(query)
    distances={}
    for stack in (3,5):
        binary=es.ChromaCrossSimilarity(frameStackSize=stack,frameStackStride=1,
                                        binarizePercentile=.095,oti=True)(query_hpcp,reference_hpcp)
        score_matrix,distance=es.CoverSongSimilarity(
            disOnset=.5,disExtension=.5,alignmentType="chen17",distanceType="asymmetric"
        )(binary)
        distances[f"frame_stack_{stack}"]=float(distance)
    distance=float(np.median(list(distances.values())))
    # Transparent display mapping: distance .10 -> 70, .20 -> 40.
    # These boundaries are preliminary and must be calibrated on a larger set.
    score=float(np.clip(100-300*distance,0,100))
    return {
        "method":"Essentia HPCP + OTI ChromaCrossSimilarity + chen17 Smith-Waterman",
        "distance":round(distance,6), "distance_by_stack":{k:round(v,6) for k,v in distances.items()},
        "score":round(score,2), "direction":"lower_distance_is_more_similar",
    }


def combine_evidence(cover: dict, rhythm: dict, acoustic: dict, fingerprint_score: float, config: dict) -> dict:
    """Combine complementary evidence; each term handles a different transform."""
    cover_score=float(cover["score"])
    rhythm_score=100*float(rhythm["score"])
    # dict.get evaluates its default eagerly, so a nested get is required when
    # callers provide only the MFCC trajectory score.
    acoustic_score=100*float(acoustic.get("mfcc_sequence",acoustic.get("score",0.0)))
    fingerprint=100*float(fingerprint_score)
    raw=(config["cover"]*cover_score + config["rhythm"]*rhythm_score + config["acoustic_sequence"]*acoustic_score
         + config["fingerprint"]*fingerprint)
    directionality=float(acoustic["directionality"])
    floor=float(config.get("reverse_direction_floor",.35))
    direction_penalty=min(1.0,directionality/floor) if floor>0 else 1.0
    # A match should be supported by both temporal groove and timbre trajectory.
    # This suppresses unrelated tracks that happen to align on only one domain.
    consensus_score=min(rhythm_score,acoustic_score)
    consensus_penalty=.6+.4*float(np.clip((consensus_score-30.0)/30.0,0.0,1.0))
    return {"score":round(raw*direction_penalty*consensus_penalty,2),"raw_score":round(raw,2),
            "cover_score":round(cover_score,2),"rhythm_score":round(rhythm_score,2),"acoustic_sequence_score":round(acoustic_score,2),
            "fingerprint_score":round(fingerprint,2),"directionality":round(directionality,6),
            "direction_penalty":round(direction_penalty,6),"consensus_score":round(consensus_score,2),
            "consensus_penalty":round(consensus_penalty,6),"weights":config}
