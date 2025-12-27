from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
import os.path as osp
import random, os
import cv2
#import cPickle as cp
import _pickle as cp
import scipy.signal as ssig
import scipy.stats as sstat
import pygame, pygame.locals
from pygame import freetype
#import Image
from PIL import Image
import math
from common import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2

import cv2
import numpy as np
import os

import cv2
import numpy as np
import os


# Совместимость со старым кодом на NumPy 2.x
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

import io

def read_lines_any_encoding(path, attempts=('utf-8', 'cp1251', 'latin-1')):
    last_err = None
    for enc in attempts:
        try:
            with io.open(path, 'r', encoding=enc, errors='strict') as f:
                return [l.strip() for l in f]
        except UnicodeDecodeError as e:
            last_err = e
            continue
    # последний шанс: «мягко» проглотить редкие символы
    with io.open(path, 'r', encoding=attempts[0], errors='replace') as f:
        return [l.strip() for l in f]

def sample_weighted(p_dict):
    ks = list(p_dict.keys())
    ps = np.array([float(p_dict[k]) for k in ks], dtype=float)
    if ps.sum() <= 0:
        ps = np.ones_like(ps) / len(ps)
    else:
        ps = ps / ps.sum()
    return np.random.choice(ks, p=ps)




def move_bb(bbs, t):
    """
    Translate the bounding-boxes in by t_x,t_y.
    BB : 2x4xn
    T  : 2-long np.array
    """
    return bbs + t[:,None,None]

def crop_safe(arr, rect, bbs=[], pad=0):
    """
    Обрезает 2D-массив (H,W) по прямоугольнику rect (x,y,w,h),
    с паддингом pad, и сдвигает bbox'ы (если переданы) в тех же координатах.

    arr : 2D ndarray (H,W)
    rect: pygame.Rect или (x,y,w,h) или (x0,y0,x1,y1)
    bbs : ndarray Nx4 в формате [x,y,w,h] (как у тебя в render_multiline)
    """
    import numpy as np

    a = np.asarray(arr)
    H, W = a.shape[:2]

    # --- распарсить rect ---
    x0 = y0 = x1 = y1 = None

    if rect is None:
        # fallback: кроп по ненулевым пикселям
        ys, xs = np.where(a > 0)
        if xs.size == 0:
            return a, bbs
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
    else:
        # pygame.Rect-like
        if hasattr(rect, "x") and hasattr(rect, "y") and hasattr(rect, "w") and hasattr(rect, "h"):
            x0 = int(rect.x)
            y0 = int(rect.y)
            x1 = int(rect.x + rect.w)
            y1 = int(rect.y + rect.h)
        else:
            r = np.asarray(rect).ravel().tolist()
            if len(r) == 4:
                a0, b0, c0, d0 = r
                # пробуем (x,y,w,h)
                if c0 >= 0 and d0 >= 0 and (a0 + c0) > a0 and (b0 + d0) > b0:
                    x0 = int(a0); y0 = int(b0); x1 = int(a0 + c0); y1 = int(b0 + d0)
                else:
                    # (x0,y0,x1,y1)
                    x0 = int(a0); y0 = int(b0); x1 = int(c0); y1 = int(d0)
            else:
                # fallback: кроп по ненулевым
                ys, xs = np.where(a > 0)
                if xs.size == 0:
                    return a, bbs
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1

    # --- pad + clamp ---
    pad = int(max(0, pad))
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)

    if x1 <= x0 or y1 <= y0:
        return a, bbs

    out = a[y0:y1, x0:x1]

    # --- сдвинуть bbox'ы ---
    if bbs is not None and len(bbs) > 0:
        b = np.asarray(bbs).copy()
        # ожидаем Nx4: x,y,w,h
        if b.ndim == 2 and b.shape[1] >= 4:
            b[:, 0] -= x0
            b[:, 1] -= y0
        bbs = b

    return out, bbs





class BaselineState(object):
    curve = lambda this, a: lambda x: a*x*x
    differential = lambda this, a: lambda x: 2*a*x
    a = [0.50, 0.05]

    def get_sample(self):
        """
        Returns the functions for the curve and differential for a and b
        """
        sgn = 1.0
        if np.random.rand() < 0.5:
            sgn = -1

        a = self.a[1]*np.random.randn() + sgn*self.a[0]
        return {
            'curve': self.curve(a),
            'diff': self.differential(a),
        }

class RenderFont(object):
    """
    Outputs a rasterized font sample.
        Output is a binary mask matrix cropped closesly with the font.
        Also, outputs ground-truth bounding boxes and text string
    """

    def __init__(self, data_dir='data'):
        # distribution over the type of text:
        self.p_text = {
            'WORD': 1.0,
            'LINE': 0.0,
            'PARA': 0.0,
        }

        ## TEXT PLACEMENT PARAMETERS:
        self.f_shrink = 0.90
        self.max_shrink_trials = 5
        self.min_nchar = 2
        self.min_font_h = 16
        self.max_font_h = 500
        self.p_flat = 0.0

        # пробел между словами (оставим как было)
        self.word_gap_px = 8
        self.char_gap_rel = 0.08    # чуть больше расстояние между буквами
        self.char_gap_px = 0        # можно 0..1
        self.text_widen_scale = 1.0 # не растягивать

        # --- жирность / морфология ---
        # По умолчанию = 0 (никакой "жирности" — чтобы буквы не склеивались)
        self.stroke_px = 1
        self.stroke_mode = "edge"   # важно: edge, не dilate

        # curved baseline:
        self.p_curved = 0
        self.baselinestate = 0.05

        # text-source : gets english text:
        self.text_source = TextSource(
            min_nchar=self.min_nchar,
            fn=osp.join(data_dir, 'newsgroup/newsgroup.txt')
        )

        # get font-state object:
        self.font_state = FontState(data_dir)

        pygame.init()

    def render_multiline(self, font, text):
        """
        Рендерит multiline текст на pygame surface и возвращает:
        surf_arr : (H,W) uint8 (0..255) маска (alpha)
        words    : строка (нормализованные пробелы)
        bbs      : Nx4 [x,y,w,h] bbox каждого символа
        """
        import numpy as np
        import pygame

        lines = str(text).split('\n')
        lengths = [len(l) for l in lines] if lines else [0]

        line_spacing = font.get_sized_height() + 1

        # размер поверхности
        line_bounds = font.get_rect(lines[np.argmax(lengths)] if lines else "O")
        fsize = (int(round(2.0 * line_bounds.width)), int(round(1.25 * line_spacing * max(1, len(lines)))))

        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

        bbs = []
        space = font.get_rect('O')
        x, y = 0, 0

        for l in lines:
            x = 0
            y += line_spacing

            for ch in l:
                if ch.isspace():
                    x += space.width
                else:
                    ch_bounds = font.render_to(surf, (x, y), ch)
                    # привести rect к глобальным координатам поверхности
                    ch_bounds.x = x + ch_bounds.x
                    ch_bounds.y = y - ch_bounds.y
                    x += ch_bounds.width
                    bbs.append(np.array([ch_bounds.x, ch_bounds.y, ch_bounds.w, ch_bounds.h], dtype=np.int32))

        # если ничего не нарисовалось
        if len(bbs) == 0:
            alpha = pygame.surfarray.pixels_alpha(surf).swapaxes(0, 1).copy()
            return alpha, ' '.join(str(text).split()), np.zeros((0, 4), dtype=np.int32)

        # union rect по bbox'ам
        x0 = int(min(bb[0] for bb in bbs))
        y0 = int(min(bb[1] for bb in bbs))
        x1 = int(max(bb[0] + bb[2] for bb in bbs))
        y1 = int(max(bb[1] + bb[3] for bb in bbs))
        rect_union = (x0, y0, x1 - x0, y1 - y0)

        words = ' '.join(str(text).split())

        # IMPORTANT: сразу приводим к (H,W)
        alpha = pygame.surfarray.pixels_alpha(surf).swapaxes(0, 1).copy()

        # кроп по rect_union + pad
        bbs_np = np.asarray(bbs, dtype=np.int32)
        alpha_c, bbs_c = crop_safe(alpha, rect_union, bbs_np, pad=15)

        return alpha_c, words, bbs_c

    
    def render_sample(self, font, mask):
        """
        Places text in the "collision-free" region as indicated
        in the mask -- 255 for unsafe, 0 for safe.
        The text is rendered using FONT, the text content is TEXT.
        """
        H, W = self.robust_HW(mask)
        f_asp = self.font_state.get_aspect_ratio(font)

        max_font_h = H  # Устанавливаем высоту шрифта равной высоте изображения

        i = 0
        while i < self.max_shrink_trials and max_font_h > self.min_font_h:
            f_h_px = self.sample_font_height_px(self.min_font_h, max_font_h)
            f_h = self.font_state.get_font_size(font, f_h_px)

            max_font_h = f_h_px
            i += 1

            font.size = f_h  # Устанавливаем размер шрифта

            # Убираем ограничение на количество символов
            nchar = 100  # Временно устанавливаем фиксированное значение для количества символов

            text_type = sample_weighted(self.p_text)
            text = self.text_source.sample(nline, nchar, text_type)
            if len(text) == 0 or np.any([len(line) == 0 for line in text]):
                continue

            # Рендерим текст с фоном
            background_color = (255, 255, 255)  # Белый фон
            alpha = 0.5  # Прозрачность фона
            text_with_background = self.render_with_background(font, text, background_color, alpha)

            # НЕ ПРОВЕРЯЕМ, выходит ли текст за пределы изображения
            # Просто пропускаем, если текст не помещается в маску
            txt_arr, txt, bb = self.render_curved(font, text)
            bb = self.bb_xywh2coords(bb)

            # Размещение текста с фоном внутри маски:
            text_mask, loc, bb, _ = self.place_text([text_with_background], mask, [bb])

            if len(loc) == 0:
                continue

            text_mask = (text_mask > 0).astype(np.uint8) * 255

            if len(loc) > 0:  # Если успешно разместили текст без столкновений:

                return text_mask, loc[0], bb[0], text

        return None  # Если не удалось разместить текст

# --- ВНУТРИ class RenderFont(object): ---

    def get_glyph_advance(self, font, ch):
        """
        Возвращает горизонтальный advance для символа ch в пикселях.
        Ставит безопасные fallbacks.
        """
        try:
            pil_font = getattr(font, "pil_font", None)
            if pil_font is not None and hasattr(pil_font, "getlength"):
                return int(round(max(0.0, pil_font.getlength(ch))))
        except Exception:
            pass

        try:
            pil_font = getattr(font, "pil_font", None)
            if pil_font is not None and hasattr(pil_font, "getsize"):
                w, _ = pil_font.getsize(ch)
                return int(w)
        except Exception:
            pass

        # Фолбэк: оценка по аспекту и размеру
        try:
            ar = float(self.font_state.get_aspect_ratio(font))
        except Exception:
            ar = 1.0
        ar = max(0.4, ar)
        return max(1, int(round(ar * float(getattr(font, "size", 16)))))

    # --- ВНУТРИ class RenderFont(object): ---

    def render_curved(self, font, text,
                    char_gap_px=None, word_gap_px=None,
                    line_gap_px=None, fp_pad_px=12):
        """
        Рендерит строку(строки) на фронтопараллельной канве (PIL).
        Возвращает:
            txt_arr       : (H,W) uint8, 0 фон, 255 текст
            txt_str       : строка с \\n
            bb_char_xywh  : (N,4) int32 [x,y,w,h] по каждому символу

        ВАЖНО:
        - Нет эрозии.
        - Нет "жирности" по умолчанию (stroke_px=0).
        - Гарантия: advance >= ширина глифа + 1 (буквы не налезают).
        - "Ширина текста" регулируется char_gap_rel / char_gap_px, а при желании ещё text_widen_scale.
        """
        import numpy as np
        import cv2
        from PIL import Image, ImageDraw, ImageFont

        if char_gap_px is None:
            char_gap_px = int(getattr(self, "char_gap_px", 0))
        if word_gap_px is None:
            word_gap_px = int(getattr(self, "word_gap_px", 8))
        if line_gap_px is None:
            line_gap_px = 4

        # линии
        if isinstance(text, list):
            lines = [str(ln) for ln in text]
        else:
            lines = str(text).split("\n")

        # PIL font
        pil_font = getattr(font, "pil_font", None)
        if pil_font is None:
            pil_font = ImageFont.truetype(getattr(font, "path"), size=int(getattr(font, "size", 16)))

        base_size = int(getattr(font, "size", 16))

        # метрики строки
        try:
            ascent, descent = pil_font.getmetrics()
        except Exception:
            ascent, descent = base_size, int(0.25 * base_size)

        line_extra = int(0.25 * base_size)
        safe_line_h = int(ascent + descent + line_extra)

        # разбор строк: символы + флаг "начало слова"
        per_line_chars = []
        per_line_wbflags = []
        for ln in lines:
            ln = str(ln)
            chars, wb = [], []
            prev_space = True
            for ch in ln:
                if ch.isspace():
                    prev_space = True
                    continue
                chars.append(ch)
                wb.append(bool(prev_space))
                prev_space = False
            per_line_chars.append(chars)
            per_line_wbflags.append(wb)

        # относительный трекинг
        char_gap_rel = float(getattr(self, "char_gap_rel", 0.0))
        char_gap_rel = float(np.clip(char_gap_rel, -0.10, 0.35))

        # собираем advances и bbox-ы глифов
        line_advances = []
        line_glyph_bboxes = []
        line_widths = []
        line_heights = []

        for chars, wbflags in zip(per_line_chars, per_line_wbflags):
            advances = []
            glyph_bboxes = []

            for ch in chars:
                try:
                    l, t, r, b = pil_font.getbbox(ch)
                except Exception:
                    mask = pil_font.getmask(ch, mode="L")
                    w, h = (mask.size if hasattr(mask, "size") else (base_size, base_size))
                    l, t, r, b = 0, 0, int(w), int(h)

                l, t, r, b = int(l), int(t), int(r), int(b)
                gw = max(1, int(r - l))
                glyph_bboxes.append((l, t, r, b))

                # базовый advance
                try:
                    adv_base = float(self.get_glyph_advance(font, ch))
                except Exception:
                    adv_base = float(gw)

                adv = adv_base * (1.0 + char_gap_rel) + float(char_gap_px)

                # ✅ КРИТИЧНО: не даём следующей букве налезть на текущую
                adv = max(adv, float(gw + 1))

                adv = max(1, int(round(adv)))
                advances.append(adv)

            # ширина строки с учётом word_gap_px
            if glyph_bboxes:
                min_left = min(bb[0] for bb in glyph_bboxes)
                left_comp = max(0, -min_left)

                wsum = int(left_comp)
                for j, adv in enumerate(advances):
                    if j > 0 and wbflags[j]:
                        wsum += int(word_gap_px)
                    wsum += int(adv)
                line_w = int(wsum)
            else:
                line_w = 0

            line_advances.append(advances)
            line_glyph_bboxes.append(glyph_bboxes)
            line_widths.append(line_w)
            line_heights.append(int(safe_line_h))

        # канва
        max_line_w = max(line_widths) if line_widths else 1
        total_w = max(1, int(max_line_w) + 2 * int(fp_pad_px))

        total_h = 2 * int(fp_pad_px)
        for i, lh in enumerate(line_heights):
            total_h += int(lh)
            if i + 1 < len(line_heights):
                total_h += int(line_gap_px)
        total_h = max(1, int(total_h))

        canvas = Image.new("L", (total_w, total_h), 0)
        draw = ImageDraw.Draw(canvas)

        # рендер + bbox
        bb_list = []
        y_cursor = int(fp_pad_px)

        for line_idx, (chars, wbflags, advances, glyph_bboxes) in enumerate(
            zip(per_line_chars, per_line_wbflags, line_advances, line_glyph_bboxes)
        ):
            x_cursor = int(fp_pad_px)
            line_h = int(line_heights[line_idx])

            if glyph_bboxes:
                min_left = min(bb[0] for bb in glyph_bboxes)
                min_top  = min(bb[1] for bb in glyph_bboxes)
            else:
                min_left, min_top = 0, 0

            base_x_shift = max(0, -int(min_left))
            base_y_shift = max(0, -int(min_top))

            for j, (ch, adv, (l, t, r, b)) in enumerate(zip(chars, advances, glyph_bboxes)):
                if j > 0 and wbflags[j]:
                    x_cursor += int(word_gap_px)

                gw = max(1, int(r - l))

                # хотим, чтобы bbox глифа начинался в (x_cursor + base_x_shift)
                glyph_x0 = int(x_cursor + base_x_shift)
                glyph_y0 = int(y_cursor + base_y_shift)

                # позиция рисования (компенсируем l,t)
                draw_x = int(glyph_x0 - l)
                draw_y = int(glyph_y0 - t)

                draw.text((draw_x, draw_y), ch, fill=255, font=pil_font)

                # bbox на всю высоту строки (как у тебя было), ширина = ширина глифа
                bb_list.append([int(glyph_x0), int(y_cursor), int(gw), int(max(1, line_h))])

                x_cursor += int(adv)

            y_cursor += int(line_h)
            if line_idx + 1 < len(line_heights):
                y_cursor += int(line_gap_px)

        # numpy
        txt_arr = np.array(canvas, dtype=np.uint8)

        # бинаризация: лучше OTSU (антиалиас не превращается в "толстый" контур)
        if txt_arr.size:
            _, txt_bin = cv2.threshold(txt_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            txt_bin = txt_arr

        # --- опциональная "жирность", но БЕЗ эрозии ---
        stroke = int(getattr(self, "stroke_px", 0))
        stroke_mode = str(getattr(self, "stroke_mode", "edge")).lower()

        # по умолчанию для больших размеров отключаем stroke (читаемость важнее)
        if base_size >= 26:
            stroke = 0

        if stroke > 0 and txt_bin.size and stroke_mode != "none":
            k = 2 * stroke + 1
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

            if stroke_mode == "dilate":
                # агрессивно — может склеивать дырки
                txt_bin = cv2.dilate(txt_bin, kern, iterations=1)
            else:
                # ✅ бережный режим: утолщаем границы, дырки живут лучше
                edge = cv2.morphologyEx(txt_bin, cv2.MORPH_GRADIENT, kern)
                edge = cv2.dilate(edge, kern, iterations=1)
                txt_bin = cv2.bitwise_or(txt_bin, edge)

        # --- "ширина текста" (горизонтальное растяжение) ---
        widen = float(getattr(self, "text_widen_scale", 1.0))
        if widen is None:
            widen = 1.0
        widen = float(np.clip(widen, 0.7, 1.6))

        if txt_bin.size and abs(widen - 1.0) > 1e-3:
            H, W = txt_bin.shape[:2]
            newW = max(1, int(round(W * widen)))
            txt_bin = cv2.resize(txt_bin, (newW, H), interpolation=cv2.INTER_NEAREST)

            # поправим bbox-ы по X
            for i in range(len(bb_list)):
                bb_list[i][0] = int(round(bb_list[i][0] * widen))
                bb_list[i][2] = max(1, int(round(bb_list[i][2] * widen)))

        txt_str = "\n".join(lines)
        bb_char_xywh = (
            np.array(bb_list, dtype=np.int32)
            if len(bb_list) else np.zeros((0, 4), dtype=np.int32)
        )

        return txt_bin, txt_str, bb_char_xywh

    def get_nline_nchar(self, mask_size, font_height, font_width):
        """
        Returns the maximum number of lines and characters which can fit
        in the MASK_SIZED image.
        """
        H, W = mask_size
        nline = int(np.ceil(H / (2 * font_height)))
        nchar = int(np.floor(W / font_width))
        return nline, nchar

    

    def place_text(self, text_arrs, back_arr, bbs):
        """
        Упрощённая и быстрая версия размещения текста.

        Сохраняет ту же сигнатуру и формат результата, что и оригинал:
            out_arr, locs, bbs, order

        text_arrs : список 2D-массивов (маски текста)
        back_arr  : 2D-массив маски/фона (используем только размер)
        bbs       : список матриц 2x4xn с координатами bbox`ов
        """
        H, W = back_arr.shape[:2]

        locs = [None] * len(text_arrs)
        out_arr = np.zeros_like(back_arr)

        for i, ta in enumerate(text_arrs):
            if ta is None:
                continue

            # ожидаем 2D-маску текста
            h, w = ta.shape[:2]
            if h <= 0 or w <= 0:
                continue

            # гарантируем, что не вылезем за границы back_arr
            h_use = min(h, H)
            w_use = min(w, W)

            max_y = max(0, H - h_use)
            max_x = max(0, W - w_use)

            y0 = np.random.randint(0, max_y + 1) if max_y > 0 else 0
            x0 = np.random.randint(0, max_x + 1) if max_x > 0 else 0

            loc = np.array([y0, x0], dtype=np.int32)
            locs[i] = loc

            # сдвигаем bbox (2x4xn) на (x0, y0)
            if bbs[i] is not None:
                shift = np.array([x0, y0], dtype=np.float32)[:, None, None]
                bbs[i] = bbs[i] + shift

            # вставляем текст в out_arr
            sub = ta[:h_use, :w_use].astype(out_arr.dtype)
            out_arr[y0:y0 + h_use, x0:x0 + w_use] = np.maximum(
                out_arr[y0:y0 + h_use, x0:x0 + w_use],
                sub,
            )

            # обновляем back_arr как маску занятости (как и раньше)
            back_arr[y0:y0 + h_use, x0:x0 + w_use] = np.maximum(
                back_arr[y0:y0 + h_use, x0:x0 + w_use],
                (sub > 0).astype(back_arr.dtype) * 255,
            )

        order = np.arange(len(text_arrs), dtype=np.int32)
        return out_arr, locs, bbs, order



    def robust_HW(self,mask):
        m = mask.copy()
        m = (~mask).astype('float')/255
        rH = np.median(np.sum(m,axis=0))
        rW = np.median(np.sum(m,axis=1))
        return rH,rW

    def sample_font_height_px(self,h_min,h_max):
        if np.random.rand() < self.p_flat:
            rnd = np.random.rand()
        else:
            rnd = np.random.beta(2.0,2.0)

        h_range = h_max - h_min
        f_h = np.floor(h_min + h_range*rnd)
        return f_h

    def bb_xywh2coords(self,bbs):
        """
        Takes an nx4 bounding-box matrix specified in x,y,w,h
        format and outputs a 2x4xn bb-matrix, (4 vertices per bb).
        """
        n,_ = bbs.shape
        coords = np.zeros((2,4,n))
        for i in range(n):
            coords[:,:,i] = bbs[i,:2][:,None]
            coords[0,1,i] += bbs[i,2]
            coords[:,2,i] += bbs[i,2:4]
            coords[1,3,i] += bbs[i,3]
        return coords


    def render_sample(self, font, mask):
        """
        Places text in the "collision-free" region as indicated
        in the mask -- 255 for unsafe, 0 for safe.
        The text is rendered using FONT, the text content is TEXT.
        """
        H, W = self.robust_HW(mask)
        f_asp = self.font_state.get_aspect_ratio(font)

        # Убираем ограничение на размер шрифта
        max_font_h = H  # Устанавливаем высоту шрифта равной высоте изображения

        i = 0
        while i < self.max_shrink_trials and max_font_h > self.min_font_h:
            f_h_px = self.sample_font_height_px(self.min_font_h, max_font_h)
            f_h = self.font_state.get_font_size(font, f_h_px)

            max_font_h = f_h_px 
            i += 1

            font.size = f_h  # Устанавливаем размер шрифта

            # Убираем ограничение на количество символов
            nchar = 100  # Временно устанавливаем фиксированное значение

            text_type = sample_weighted(self.p_text)
            text = self.text_source.sample(nline, nchar, text_type)
            if len(text) == 0 or np.any([len(line) == 0 for line in text]):
                continue

            # Рендерим текст с фоном
            background_color = (255, 255, 255)  # Белый фон
            alpha = 0.5  # Прозрачность фона
            text_with_background = self.render_with_background(font, text, background_color, alpha)


            txt_arr, txt, bb = self.render_curved(font, text)
            bb = self.bb_xywh2coords(bb)

            # Размещение текста с фоном внутри маски:
            text_mask, loc, bb, _ = self.place_text([text_with_background], mask, [bb])

            if len(loc) == 0:
                continue

            text_mask = (text_mask > 0).astype(np.uint8) * 255

            if len(loc) > 0:  # Если успешно разместили текст без столкновений:
                return text_mask, loc[0], bb[0], text

        return None  # Если не удалось разместить текст



    def visualize_bb(self, text_arr, bbs):
        ta = text_arr.copy()
        for r in bbs:
            cv.rectangle(ta, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), color=128, thickness=1)
        plt.imshow(ta,cmap='gray')
        plt.show()


class FontState(object):
    """
    Defines the random state of the font rendering  
    """
    size = [50, 10]  # normal dist mean, std
    underline = 0.05
    strong = 0.5
    oblique = 0.2
    wide = 0.5
    strength = [0.05, 0.1]  # uniform dist in this interval
    underline_adjustment = [1.0, 2.0]  # normal dist mean, std
    kerning = [2, 5, 0, 20]  # beta distribution alpha, beta, offset, range (mean is a/(a+b))
    border = 0.25
    random_caps = -1 ## don't recapitalize : retain the capitalization of the lexicon
    capsmode = [str.lower, str.upper, str.capitalize]  # lower case, upper case, proper noun
    curved = 0.2
    random_kerning = 0.2
    random_kerning_amount = 0.1

    def __init__(self, data_dir='data'):

        char_freq_path = osp.join(data_dir, 'models/char_freq.cp')        
        font_model_path = osp.join(data_dir, 'models/font_px2pt.cp')

        # get character-frequencies in the English language:
        with open(char_freq_path,'rb') as f:
            #self.char_freq = cp.load(f)
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load()
            self.char_freq = p

        # get the model to convert from pixel to font pt size:
        with open(font_model_path,'rb') as f:
            #self.font_model = cp.load(f)
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load()
            self.font_model = p
            
        # get the names of fonts to use:
        self.FONT_LIST = osp.join(data_dir, 'fonts/fontlist.txt')
        self.fonts = [os.path.join(data_dir,'fonts',f.strip()) for f in open(self.FONT_LIST)]


    def get_aspect_ratio(self, font, size=None):
        """
        Returns the median aspect ratio of each character of the font.
        """
        if size is None:
            size = 12 # doesn't matter as we take the RATIO
        chars = ''.join(self.char_freq.keys())
        w = np.array(self.char_freq.values())

        # get the [height,width] of each character:
        try:
            sizes = font.get_metrics(chars,size)
            good_idx = [i for i in range(len(sizes)) if sizes[i] is not None]
            sizes,w = [sizes[i] for i in good_idx], w[good_idx]
            sizes = np.array(sizes).astype('float')[:,[3,4]]        
            r = np.abs(sizes[:,1]/sizes[:,0]) # width/height
            good = np.isfinite(r)
            r = r[good]
            w = w[good]
            w /= np.sum(w)
            r_avg = np.sum(w*r)
            return r_avg
        except:
            return 1.0

    def get_font_size(self, font, font_size_px):
        """
        Returns the font-size which corresponds to FONT_SIZE_PX pixels font height.
        """
        m = self.font_model[font.name]
        return m[0]*font_size_px + m[1] #linear model



    def sample(self):
        """
        Samples from the font state distribution.
        Подчёркивание (underline) отключено полностью.
        """
        # underline_adjustment: нормальный clamp в [-2, 2]
        ua = self.underline_adjustment[1] * np.random.randn() + self.underline_adjustment[0]
        ua = float(np.clip(ua, -2.0, 2.0))

        return {
            'font': self.fonts[int(np.random.randint(0, len(self.fonts)))],
            'size': self.size[1] * np.random.randn() + self.size[0],

            # ✅ underline выключен навсегда:
            'underline': False,
            'underline_adjustment': ua,

            'strong': np.random.rand() < self.strong,
            'oblique': np.random.rand() < self.oblique,
            'strength': (self.strength[1] - self.strength[0]) * np.random.rand() + self.strength[0],
            'char_spacing': int(self.kerning[3] * (np.random.beta(self.kerning[0], self.kerning[1])) + self.kerning[2]),
            'border': np.random.rand() < self.border,
            'random_caps': np.random.rand() < self.random_caps,
            'capsmode': random.choice(self.capsmode),
            'curved': np.random.rand() < self.curved,
            'random_kerning': np.random.rand() < self.random_kerning,
            'random_kerning_amount': self.random_kerning_amount,
        }

    def init_font(self, fs):
        """
        Initializes a pygame font.
        Подчёркивание (underline) отключено полностью, даже если где-то ещё fs содержит underline=True.
        """
        font = freetype.Font(fs['font'], size=fs['size'])

        # ✅ жёстко выключаем underline:
        font.underline = False
        # underline_adjustment оставим, но он ни на что не повлияет при underline=False
        font.underline_adjustment = fs.get('underline_adjustment', 0.0)

        font.strong = fs['strong']
        font.oblique = fs['oblique']
        font.strength = fs['strength']

        # остальное как было
        font.antialiased = False
        font.origin = True
        return font


class TextSource(object):
    """
    Provides text for words, paragraphs, sentences.
    """
    def __init__(self, min_nchar, fn):
        """
        TXT_FN : path to file containing text data.
        """
        self.min_nchar = min_nchar
        self.fdict = {'WORD':self.sample_word,
                      'LINE':self.sample_line,
                      'PARA':self.sample_para}

        self.txt = read_lines_any_encoding(fn)

        # distribution over line/words for LINE/PARA:
        self.p_line_nline = np.array([0.85, 0.10, 0.05])
        self.p_line_nword = [4,3,12]  # normal: (mu, std)
        self.p_para_nline = [1.0,1.0]#[1.7,3.0] # beta: (a, b), max_nline
        self.p_para_nword = [1.7,3.0,10] # beta: (a,b), max_nword

        # probability to center-align a paragraph:
        self.center_para = 0.5


    def check_symb_frac(self, txt, f=0.35):
        """
        T/F return : T iff fraction of symbol/special-charcters in
                     txt is less than or equal to f (default=0.25).
        """
        return np.sum([not ch.isalnum() for ch in txt])/(len(txt)+0.0) <= f

    def is_good(self, txt, f=0.35):
        """
        T/F return : T iff the lines in txt (a list of txt lines)
                     are "valid".
                     A given line l is valid iff:
                         1. It is not empty.
                         2. symbol_fraction > f
                         3. Has at-least self.min_nchar characters
                         4. Not all characters are i,x,0,O,-
        """
        def is_txt(l):
            char_ex = ['i','I','o','O','0','-']
            chs = [ch in char_ex for ch in l]
            return not np.all(chs)

        return [ (len(l)> self.min_nchar
                 and self.check_symb_frac(l,f)
                 and is_txt(l)) for l in txt ]

    def center_align(self, lines):
        """
        PADS lines with space to center align them
        lines : list of text-lines.
        """
        ls = [len(l) for l in lines]
        max_l = max(ls)
        for i in range(len(lines)):
            l = lines[i].strip()
            dl = max_l-ls[i]
            lspace = dl//2
            rspace = dl-lspace
            lines[i] = ' '*lspace+l+' '*rspace
        return lines

    def get_lines(self, nline, nword, nchar_max, f=0.35, niter=100):
        def h_lines(niter=100):
            lines = ['']
            iter = 0
            while not np.all(self.is_good(lines,f)) and iter < niter:
                iter += 1
                line_start = np.random.choice(len(self.txt)-nline)
                lines = [self.txt[line_start+i] for i in range(nline)]
            return lines

        lines = ['']
        iter = 0
        while not np.all(self.is_good(lines,f)) and iter < niter:
            iter += 1
            lines = h_lines(niter=100)
            # get words per line:
            nline = len(lines)
            for i in range(nline):
                words = lines[i].split()
                dw = len(words)-nword[i]
                if dw > 0:
                    first_word_index = random.choice(range(dw+1))
                    lines[i] = ' '.join(words[first_word_index:first_word_index+nword[i]])

                while len(lines[i]) > nchar_max: #chop-off characters from end:
                    if not np.any([ch.isspace() for ch in lines[i]]):
                        lines[i] = ''
                    else:
                        lines[i] = lines[i][:len(lines[i])-lines[i][::-1].find(' ')].strip()
        
        if not np.all(self.is_good(lines,f)):
            return #None
        else:
            return lines

    def sample(self, nline_max,nchar_max,kind='WORD'):
        return self.fdict[kind](nline_max,nchar_max)
        
    def sample_word(self,nline_max,nchar_max,niter=100):
        rand_line = self.txt[np.random.choice(len(self.txt))]                
        words = rand_line.split()
        rand_word = random.choice(words)

        iter = 0
        while iter < niter and (not self.is_good([rand_word])[0] or len(rand_word)>nchar_max):
            rand_line = self.txt[np.random.choice(len(self.txt))]                
            words = rand_line.split()
            rand_word = random.choice(words)
            iter += 1

        if not self.is_good([rand_word])[0] or len(rand_word)>nchar_max:
            return []
        else:
            return rand_word


    def sample_line(self,nline_max,nchar_max):
        nline = nline_max+1
        while nline > nline_max:
            nline = np.random.choice([1,2,3], p=self.p_line_nline)

        # get number of words:
        nword = [self.p_line_nword[2]*sstat.beta.rvs(a=self.p_line_nword[0], b=self.p_line_nword[1])
                 for _ in range(nline)]
        nword = [max(1,int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        if lines is not None:
            return '\n'.join(lines)
        else:
            return []

    def sample_para(self,nline_max,nchar_max):
        # get number of lines in the paragraph:
        nline = nline_max*sstat.beta.rvs(a=self.p_para_nline[0], b=self.p_para_nline[1])
        nline = max(1, int(np.ceil(nline)))

        # get number of words:
        nword = [self.p_para_nword[2]*sstat.beta.rvs(a=self.p_para_nword[0], b=self.p_para_nword[1])
                 for _ in range(nline)]
        nword = [max(1,int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        if lines is not None:
            # center align the paragraph-text:
            if np.random.rand() < self.center_para:
                lines = self.center_align(lines)
            return '\n'.join(lines)
        else:
            return []