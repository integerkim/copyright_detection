from audio.code.cover_similarity import combine_evidence

CFG={"cover":.6,"rhythm":.15,"acoustic_sequence":.15,"fingerprint":.1,"reverse_direction_floor":.35}

def evidence(cover,rhythm,mfcc,fingerprint=0,direction=.5):
    return combine_evidence({"score":cover},{"score":rhythm/100},{"mfcc_sequence":mfcc/100,"directionality":direction},fingerprint/100,CFG)["score"]

def test_composite_expected_order():
    assert evidence(84,100,100,100,1) > evidence(84,90,75,1,.55) > evidence(50,72,62,1,.5) > evidence(27,53,32,1,.5)

def test_reverse_direction_is_penalized():
    assert evidence(53,66,81,2,.1) < 20
