import numpy as np
from audio.code.evidence_feature import build_high_level,extract_low_level,rhythm_similarity

def test_low_and_high_are_distinct_representations():
    sr=22050; t=np.arange(sr*4)/sr; audio=np.sin(2*np.pi*220*t).astype(np.float32)
    low=extract_low_level(audio,sr); high=build_high_level(low)
    assert "onset_envelope" in low and "groove_sequence" in high
    assert "onset_envelope" in low and "mfcc_frames" in low
    assert "groove_sequence" in high and "timbre_trajectory" in high
    assert "f0_hz" not in low and "structure_recurrence" not in high
    assert "mfcc_frames" in low and "timbre_trajectory" in high

def test_identical_groove_scores_high():
    groove=np.eye(4,dtype=np.float32)
    result=rhythm_similarity({"groove_sequence":groove},{"groove_sequence":groove})
    assert result["score"]>.99
