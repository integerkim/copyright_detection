from pathlib import Path
from audio.code.audio_main import source_tag

def test_cache_key_depends_on_input_name():
    assert source_tag(0,Path("a.wav")) != source_tag(0,Path("b.wav"))
    assert source_tag(0,Path("my audio.mp3")) == "query_0000_my_audio"
