def test_cli_imports():
    from audio.code.audio_main import main
    assert callable(main)
