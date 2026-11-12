"""Image validation and image-analysis tasks.

Uploaded or pasted images are NEVER trusted:
  * size is checked before and after decoding, pixel count is capped (decompression bombs);
  * the real format is detected by Pillow (the declared MIME type is ignored);
  * the image is re-encoded as a clean JPEG, which strips EXIF/GPS metadata and any appended data;
  * large images are scaled down so they fit provider limits and stay fast.

`source` ("upload", "screenshot", "screen_stream") is carried through the API so continuous
screen understanding can later feed frames through this same service.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import io
import re
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageOps, UnidentifiedImageError

from ai.base import ImagePart

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}

TASK_INSTRUCTIONS = {
    "auto": "Work out what the image shows (equation, handwriting, diagram, graph, table, screenshot, photo) and answer the user's request about it.",
    "equation": ("Transcribe every mathematical expression exactly and give it in LaTeX. Then solve, simplify or explain as asked. "
                 "Verify results with the maths tool when it is available. If part of the handwriting is ambiguous, say which reading you used."),
    "diagram": ("Describe the diagram's structure first: components, labels, arrows, units and relationships. Then interpret it "
                "scientifically and answer the user's question."),
    "chart": ("Read the chart carefully: type, title, axes with units, series and legend, notable values and trends. "
              "Say clearly when a value is estimated from the picture. Do not invent data points."),
    "screenshot": ("Describe what the screenshot shows (application, text, code, error messages, UI state), then help with the "
                   "user's request. Quote on-screen text exactly when it matters."),
    "text": "Transcribe the text faithfully, preserving structure (headings, lists, tables), then answer the user's request.",
}


class ImageValidationError(ValueError):
    """The image cannot be accepted. `status` is the HTTP status to return."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


@dataclass
class PreparedImage:
    name: str
    mime_type: str
    data_b64: str
    width: int
    height: int
    size_bytes: int        # size of the ORIGINAL upload
    sha256: str            # hash of the original, handy for logs and dedupe

    def part(self) -> ImagePart:
        return ImagePart(self.mime_type, self.data_b64)


@dataclass
class RawImage:
    name: str
    data: bytes


def task_instruction(task: str) -> tuple[str, Optional[str]]:
    """Return (instruction, warning). Unknown tasks fall back to 'auto'."""
    if task in TASK_INSTRUCTIONS:
        return TASK_INSTRUCTIONS[task], None
    return TASK_INSTRUCTIONS["auto"], f"Unknown image task '{task}'; 'auto' was used."


class ImageService:
    def __init__(self, settings):
        self.settings = settings
        Image.MAX_IMAGE_PIXELS = settings.image_max_pixels  # Pillow raises on 2x this value

    # --- decoding -------------------------------------------------------------------
    def decode_base64(self, data: str) -> bytes:
        if data.startswith("data:"):
            _, _, data = data.partition(",")
        data = re.sub(r"\s+", "", data)
        if len(data) * 3 // 4 > self.settings.image_max_bytes + 4:
            raise ImageValidationError(f"Image is larger than {self.settings.image_max_bytes // (1024 * 1024)} MB.", 413)
        try:
            return base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            raise ImageValidationError("Image data is not valid base64.") from None

    def _process(self, raw: bytes, name: str) -> PreparedImage:
        if not raw:
            raise ImageValidationError("The image is empty.")
        if len(raw) > self.settings.image_max_bytes:
            raise ImageValidationError(f"Image is larger than {self.settings.image_max_bytes // (1024 * 1024)} MB.", 413)
        try:
            with Image.open(io.BytesIO(raw)) as probe:
                fmt = probe.format
                probe.verify()  # structural check; the file must be reopened afterwards
            if fmt not in ALLOWED_FORMATS:
                raise ImageValidationError("Unsupported image type. Use JPEG, PNG, WebP or GIF.", 415)
            with Image.open(io.BytesIO(raw)) as img:
                if img.width * img.height > self.settings.image_max_pixels:
                    raise ImageValidationError("The image has too many pixels.", 413)
                img = ImageOps.exif_transpose(img)  # honour phone camera rotation
                img.load()                         # first frame for GIFs
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    rgba = img.convert("RGBA")
                    flat = Image.new("RGB", rgba.size, (255, 255, 255))
                    flat.paste(rgba, mask=rgba.split()[-1])
                    img = flat
                else:
                    img = img.convert("RGB")
                side = self.settings.image_max_side
                if max(img.size) > side:
                    img.thumbnail((side, side), Image.LANCZOS)
                out = io.BytesIO()
                img.save(out, format="JPEG", quality=90, optimize=True)
                return PreparedImage(
                    name=re.sub(r"[^\w.\- ]", "_", name or "image")[:100],
                    mime_type="image/jpeg",
                    data_b64=base64.b64encode(out.getvalue()).decode("ascii"),
                    width=img.width, height=img.height,
                    size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()[:16],
                )
        except ImageValidationError:
            raise
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError) as exc:
            raise ImageValidationError("That file is not a readable image.") from exc

    # --- public API -----------------------------------------------------------------
    async def prepare_raw(self, items: list[RawImage]) -> list[PreparedImage]:
        if len(items) > self.settings.max_images:
            raise ImageValidationError(f"At most {self.settings.max_images} images per request.", 413)
        return list(await asyncio.gather(*(asyncio.to_thread(self._process, i.data, i.name) for i in items)))

    async def prepare_payloads(self, payloads) -> list[PreparedImage]:
        """Accepts ImagePayload objects from JSON requests (base64)."""
        raws = [RawImage(p.name, self.decode_base64(p.data)) for p in payloads]
        return await self.prepare_raw(raws)
