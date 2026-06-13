"""Explicit bridge between scene rendering and text utilities.

`synthtext.rendering.text_utils` owns text corpora, font sampling and low-level text masks.
`RendererV3` owns scene regions, geometry and final image composition.  This
module keeps the contract between those two sides small and named.
"""

from collections import deque
from dataclasses import dataclass
import re

import numpy as np

from . import text_utils as tu


@dataclass
class FontContext:
    """A sampled font plus the metrics RendererV3 needs for placement."""

    state: dict
    font: object
    aspect_ratio: float


@dataclass
class LayoutText:
    """Text sampled for a candidate placement."""

    text: str
    lang: str


class TextRenderingService:
    """
    Narrow public interface from scene rendering into `text_utils.RenderFont`.

    The service intentionally exposes verbs used by the renderer instead of
    leaking `RenderFont.font_state`, `RenderFont.text_source` and queue details.
    """

    def __init__(self, data_dir):
        self.renderer = tu.RenderFont(data_dir)
        self._word_queue = deque()

    def sample_font(self) -> FontContext:
        fs = self.renderer.font_state.sample()
        font = self.renderer.font_state.init_font(fs)
        aspect_ratio = self.renderer.font_state.get_aspect_ratio(font)
        return FontContext(fs, font, self._sanitize_aspect_ratio(aspect_ratio))

    def estimate_layout_capacity(self, font_height_px, aspect_ratio, mask_size=(128, 512),
                                 min_chars=6, fallback=(1, 12)):
        try:
            nline_raw, nchar_raw = self.renderer.get_nline_nchar(
                mask_size,
                float(font_height_px),
                float(font_height_px) * self._sanitize_aspect_ratio(aspect_ratio),
            )
        except Exception:
            nline_raw, nchar_raw = fallback

        nline = max(1, int(nline_raw or fallback[0]))
        nchar = fallback[1] if nchar_raw is None or int(nchar_raw) < int(min_chars) else int(nchar_raw)
        return nline, nchar

    def set_font_size_px(self, font, font_height_px):
        font.size = self.renderer.font_state.get_font_size(font, float(font_height_px))
        return font.size

    def sample_layout_text(self, nline, nchar, min_word_len=4, max_retries=20) -> LayoutText:
        """
        Return one word-like token and its language.

        Internally this keeps the old queue behavior: a sampled corpus fragment
        is tokenized once, then words are consumed one by one by future calls.
        """
        min_word_len = int(min_word_len)

        while self._word_queue:
            word, lang = self._word_queue.popleft()
            if len(word) >= min_word_len:
                return LayoutText(word, lang)

        last_raw_text = ""
        last_lang = "unk"

        for _ in range(int(max_retries)):
            kind = self._sample_text_kind()
            raw_obj, last_lang = self._sample_raw_text(nline, nchar, kind)
            last_raw_text = self._normalize_raw_text(raw_obj)
            if not last_raw_text:
                continue

            words = self._tokenize(last_raw_text)
            if not words:
                continue

            self._word_queue = deque((word, last_lang) for word in words)
            while self._word_queue:
                word, lang = self._word_queue.popleft()
                if len(word) >= min_word_len:
                    return LayoutText(word, lang)

        if last_raw_text:
            words = self._tokenize(last_raw_text)
            if words:
                return LayoutText(words[0], last_lang)

        return LayoutText("text", "unk")

    def clear_word_queue(self):
        self._word_queue.clear()

    def render_curved(self, font, text, **kwargs):
        return self.renderer.render_curved(font, text, **kwargs)

    def render_sample(self, font, mask):
        return self.renderer.render_sample(font, mask)

    @staticmethod
    def _sanitize_aspect_ratio(value):
        try:
            value = float(value)
        except Exception:
            return 1.0
        if not np.isfinite(value) or value <= 1e-6:
            return 1.0
        return value

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", str(text))

    @staticmethod
    def _normalize_raw_text(raw_obj):
        if isinstance(raw_obj, list):
            return " ".join(str(x).strip() for x in raw_obj if str(x).strip())
        if isinstance(raw_obj, str):
            return raw_obj.strip()
        return ""

    def _sample_text_kind(self):
        try:
            return tu.sample_weighted(self.renderer.p_text)
        except Exception:
            return None

    def _sample_raw_text(self, nline, nchar, kind):
        try:
            return self.renderer.text_source.sample(nline, nchar, kind, return_lang=True)
        except Exception:
            return None, "unk"
