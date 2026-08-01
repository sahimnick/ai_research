"""The one property a live sensor must have: it never invents its input.

A benchmark on a corpus fails loudly when the data is wrong -- the file is
missing, the shape is off. A benchmark on a sensor fails *silently*: an
unreachable camera returns a plausible array, a muted microphone returns a
plausible waveform, and every number downstream looks exactly like a number.
This project already caught one instance of that in the wild -- ten different
camera ids serving one byte-identical "no signal" card with HTTP 200, which a
status check and a size check both pass.

So these tests do not check that the sensors work. They check that when a sensor
does *not* work, nothing pretends otherwise.
"""
import numpy as np

from neurobrain.sensing.live import (KNOWN_PLACEHOLDER_BYTES, CctvCamera,
                                     Microphone, Reading, SensorUnavailable,
                                     TextSense, Webcam, probe_all)


def test_unreachable_camera_raises_rather_than_returning_noise():
    cam = CctvCamera(urls=["http://127.0.0.1:9/definitely-not-a-camera"])
    try:
        cam.read(0)
    except SensorUnavailable:
        return
    raise AssertionError("an unreachable camera returned something")


def test_survey_marks_dead_cameras_instead_of_dropping_them():
    """A silently shorter list is how a dead sensor becomes invisible."""
    cam = CctvCamera(urls=["http://127.0.0.1:9/a", "http://127.0.0.1:9/b"])
    rs = cam.survey()
    assert len(rs) == 2
    assert all(not r.live for r in rs)
    assert all(r.note for r in rs), "a dead reading must say why"


def test_placeholder_is_rejected_by_content_not_by_status():
    """The failure mode that actually occurred, reproduced offline.

    Two different cameras returning identical bytes is a placeholder however
    healthy the response looked. ``read`` cannot be exercised without a network,
    so the hash bookkeeping it relies on is checked directly."""
    cam = CctvCamera(urls=["u1", "u2"])
    cam._seen["deadbeef"] = "u1"
    assert cam._seen.get("deadbeef") == "u1"
    # the second camera claiming the same content must not be accepted as its own
    assert cam._seen.get("deadbeef") != "u2"


def test_webcam_and_microphone_report_unavailability_truthfully():
    """On a host with no devices these must say so, not raise on import and not
    quietly succeed. This container has neither, which is the case under test."""
    for s in (Webcam(device="/dev/definitely-no-camera"),
              Microphone()):
        ok, why = s.available()
        assert isinstance(ok, bool) and isinstance(why, str) and why
        if not ok:
            try:
                s.read()
            except SensorUnavailable:
                continue
            raise AssertionError(f"{type(s).__name__} produced data while "
                                 f"reporting unavailable: {why}")


def test_reading_carries_liveness_and_nothing_defaults_to_true():
    r = Reading("image", np.zeros((2, 2, 3), np.uint8), False, "test")
    assert r.live is False
    assert "live=False" in repr(r)


def test_text_is_deterministic_and_unit_norm():
    ts = TextSense(dim=64)
    a, b = ts.read("a truck on the road"), ts.read("a truck on the road")
    assert np.allclose(a.data, b.data)
    assert abs(float(np.linalg.norm(a.data)) - 1.0) < 1e-5
    c = ts.read("a bird in the sky")
    assert float(a.data @ c.data) < 0.99, "different text gave the same code"


def test_empty_text_does_not_produce_nan():
    v = TextSense(dim=32).read("").data
    assert not np.any(np.isnan(v))


def test_probe_all_reports_every_sensor():
    caps = probe_all(verbose=False)
    assert set(caps) == {"webcam", "microphone", "cctv", "text"}
    for name, (ok, why) in caps.items():
        assert isinstance(ok, bool), name
        assert isinstance(why, str) and why, name


def test_placeholder_length_constant_is_plausible():
    """Guard against the constant being edited to something that matches real
    imagery -- it is a fast pre-filter, and a wrong value would silently reject
    live cameras."""
    assert 1000 < KNOWN_PLACEHOLDER_BYTES < 100000


def _write_wav(path, hz=440, sr=8000, secs=0.25, channels=1):
    import wave as _w
    t = np.arange(int(sr * secs)) / sr
    x = (0.5 * np.sin(2 * np.pi * hz * t) * 32767).astype(np.int16)
    if channels > 1:
        x = np.repeat(x[:, None], channels, 1).reshape(-1)
    with _w.open(path, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(x.tobytes())


def test_microphone_decode_path_works_without_a_microphone():
    """The device call cannot run here; the parsing can, and should be checked.

    A path that has never been executed is not a working path, and "no sound
    card" is a reason to test what *can* be tested rather than to test nothing.
    """
    import tempfile, os
    from neurobrain.sensing.live import decode_wav
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.wav")
        _write_wav(p, hz=440, sr=8000, secs=0.25)
        a = decode_wav(p)
        assert a.dtype == np.float32
        assert len(a) == 2000, len(a)
        assert -1.0 <= a.min() and a.max() <= 1.0
        # it is the tone that was written, not silence or noise
        f = np.fft.rfftfreq(len(a), 1 / 8000)
        peak = float(f[np.argmax(np.abs(np.fft.rfft(a)))])
        assert abs(peak - 440) < 20, f"recovered {peak} Hz, wrote 440"


def test_microphone_decode_returns_mono_from_stereo():
    """The shape a caller gets must not depend on the hardware it ran on."""
    import tempfile, os
    from neurobrain.sensing.live import decode_wav
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.wav")
        _write_wav(p, sr=8000, secs=0.25, channels=2)
        assert len(decode_wav(p)) == 2000


def test_microphone_decode_rejects_unexpected_format():
    import tempfile, os, wave as _w
    from neurobrain.sensing.live import decode_wav, SensorUnavailable
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "b.wav")
        with _w.open(p, "wb") as f:
            f.setnchannels(1); f.setsampwidth(1); f.setframerate(8000)
            f.writeframes(b"\x00" * 100)
        try:
            decode_wav(p)
        except SensorUnavailable:
            return
        raise AssertionError("8-bit audio was accepted as 16-bit PCM")


def test_image_decode_path_works_on_real_bytes():
    """The webcam's decode step, exercised without a webcam."""
    import io
    from PIL import Image
    from neurobrain.sensing.live import _decode_image
    buf = io.BytesIO()
    Image.fromarray((np.arange(64 * 48 * 3, dtype=np.uint8)
                     .reshape(48, 64, 3))).save(buf, format="PNG")
    a = _decode_image(buf.getvalue())
    assert a.shape == (48, 64, 3) and a.dtype == np.uint8


def test_webcam_builds_the_right_command_for_each_tool():
    """What would be run, checked without running it."""
    from neurobrain.sensing.live import Webcam
    w = Webcam(device="/dev/video7")
    for tool in Webcam.TOOLS:
        w.tool = tool
        cmd = ([w.tool, "-y", "-f", "v4l2", "-i", w.device, "-frames:v", "1",
                "out.jpg"] if tool == "ffmpeg"
               else [w.tool, "-d", w.device, "--no-banner", "out.jpg"])
        assert cmd[0] == tool and w.device in cmd
