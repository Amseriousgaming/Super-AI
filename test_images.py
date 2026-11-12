"""Image validation: valid formats are normalised, hostile or oversized input is refused."""
import base64
import io

import pytest
from PIL import Image

from services.image_analysis import ImageService, ImageValidationError, RawImage
from tests.fakes import make_settings, run


def png(size=(1600, 1200), mode="RGB", fmt="PNG"):
    color = (10, 120, 200, 128) if mode == "RGBA" else (10, 120, 200)
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format=fmt)
    return buf.getvalue()


def service(**kw):
    return ImageService(make_settings(image_max_side=800, image_max_bytes=2 * 1024 * 1024, image_max_pixels=5_000_000, **kw))


@pytest.mark.parametrize("fmt,mode", [("JPEG", "RGB"), ("PNG", "RGBA"), ("GIF", "RGB"), ("WEBP", "RGB")])
def test_valid_images_become_clean_jpegs(fmt, mode):
    out = run(service().prepare_raw([RawImage("x", png(mode=mode, fmt=fmt))]))[0]
    assert out.mime_type == "image/jpeg" and max(out.width, out.height) <= 800
    assert base64.b64decode(out.data_b64)[:2] == b"\xff\xd8"


@pytest.mark.parametrize("raw,status", [(b"not an image", 422), (b"", 422), (png(fmt="BMP", size=(10, 10)), 415),
                                        (png(size=(3000, 2500)), 413), (b"\xff\xd8" + b"0" * (3 * 1024 * 1024), 413)])
def test_bad_images_are_rejected(raw, status):
    with pytest.raises(ImageValidationError) as exc:
        run(service().prepare_raw([RawImage("x", raw)]))
    assert exc.value.status == status


def test_base64_handling_and_count_limit():
    svc = service()
    data_url = "data:image/png;base64," + base64.b64encode(png(size=(50, 50))).decode()
    assert svc.decode_base64(data_url)
    with pytest.raises(ImageValidationError):
        svc.decode_base64("@@@@not-base64!!")
    with pytest.raises(ImageValidationError):
        run(ImageService(make_settings(max_images=2)).prepare_raw([RawImage("a", png(size=(10, 10)))] * 3))
