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
    Быстрая "no-op" версия:
    - не обрезает изображение,
    - не трогает bbox'ы,
    - оставлена для совместимости по сигнатуре.

    Возвращает:
        arr, bbs
    """
    return arr, bbs




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
        # whether to get a single word, paragraph or a line:
        self.p_text = {
            'WORD': 1.0,
            'LINE': 0.0,
            'PARA': 0.0,
        }

        ## TEXT PLACEMENT PARAMETERS:
        self.f_shrink = 0.90
        self.max_shrink_trials = 5 # 0.9^5 ~= 0.6
        # the minimum number of characters that should fit in a mask
        # to define the maximum font height.
        self.min_nchar = 2
        self.min_font_h = 16 #px : 0.6*12 ~ 7px <= actual minimum height
        self.max_font_h = 500 #px
        self.p_flat = 0.0
        self.char_gap_px = 0     # тонкий tracking
        self.word_gap_px = 8

        # curved baseline:
        self.p_curved = 0
        self.baselinestate = 0.05

        # text-source : gets english text:
        self.text_source = TextSource(min_nchar=self.min_nchar,
                                      fn=osp.join(data_dir,'newsgroup/newsgroup.txt'))

        # get font-state object:
        self.font_state = FontState(data_dir)

        pygame.init()

    def render_multiline(self,font,text):
        """
        renders multiline TEXT on the pygame surface SURF with the
        font style FONT.
        A new line in text is denoted by \n, no other characters are 
        escaped. Other forms of white-spaces should be converted to space.

        returns the updated surface, words and the character bounding boxes.
        """
        # get the number of lines
        lines = text.split('\n')
        lengths = [len(l) for l in lines]

        # font parameters:
        line_spacing = font.get_sized_height() + 1
        
        # initialize the surface to proper size:
        line_bounds = font.get_rect(lines[np.argmax(lengths)])
        fsize = (round(2.0*line_bounds.width), round(1.25*line_spacing*len(lines)))
        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

        bbs = []
        space = font.get_rect('O')
        x, y = 0, 0
        for l in lines:
            x = 0 # carriage-return
            y += line_spacing # line-feed

            for ch in l: # render each character
                if ch.isspace(): # just shift
                    x += space.width
                else:
                    # render the character
                    ch_bounds = font.render_to(surf, (x,y), ch)
                    ch_bounds.x = x + ch_bounds.x
                    ch_bounds.y = y - ch_bounds.y
                    x += ch_bounds.width
                    bbs.append(np.array(ch_bounds))

        # get the union of characters for cropping:
        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)

        # get the words:
        words = ' '.join(text.split())

        # crop the surface to fit the text:
        bbs = np.array(bbs)
        surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=15)
        surf_arr = surf_arr.swapaxes(0,1)
        #self.visualize_bb(surf_arr,bbs)
        return surf_arr, words, bbs
    
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


    
    # Внутри class RenderFont:
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
        ar = max(0.4, float(self.font_state.get_aspect_ratio(font)))
        return max(1, int(round(ar * float(getattr(font, "size", 16)))))



    def render_curved(self, font, text,
                  char_gap_px=0, word_gap_px=8,
                  line_gap_px=None, fp_pad_px=12):
        """
        Рендерит строку(строки) на фронтопараллельной канве.
        Возвращает:
            txt_arr       : HxW uint8, 0 фон, 255 текст
            txt_str       : строка с \\n между линиями
            bb_char_xywh  : (N, 4) int32, [x, y, w, h] для каждого символа
        """
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        # 0) Автоматический подбор line_gap_px
        if line_gap_px is None:
            line_gap_px = 4

        # 0) Линии источника
        if isinstance(text, list):
            lines = [str(ln) for ln in text]
        else:
            lines = str(text).split("\n")

        # 1) Подготовим PIL-шрифт
        pil_font = getattr(font, "pil_font", None)
        if pil_font is None:
            pil_font = ImageFont.truetype(
                getattr(font, "path"),
                size=getattr(font, "size", 16)
            )

        base_size = getattr(font, "size", 16)
        ascent, descent = pil_font.getmetrics()
        line_extra = int(0.25 * base_size)
        safe_line_h = int(ascent + descent + line_extra)

        # 2) Разбор по строкам
        per_line_chars   = []
        per_line_wbflags = []
        for ln in lines:
            chars, flags = [], []
            prev_space = True
            for ch in ln:
                if ch.isspace():
                    prev_space = True
                    continue
                chars.append(ch)
                flags.append(prev_space)
                prev_space = False
            per_line_chars.append(chars)
            per_line_wbflags.append(flags)

        # 3) Метрики по строкам
        line_advances      = []
        line_glyph_bboxes  = []
        line_widths        = []
        line_heights       = []

        char_gap_rel = getattr(self, "char_gap_rel", -0.15)

        for chars, wbflags in zip(per_line_chars, per_line_wbflags):
            advances = []
            glyph_bboxes = []

            for ch in chars:
                # bbox глифа
                try:
                    l, t, r, b = pil_font.getbbox(ch)
                except Exception:
                    mask = pil_font.getmask(ch, mode="L")
                    if hasattr(mask, "size"):
                        w, h = mask.size
                    else:
                        w, h = base_size, base_size
                    l, t, r, b = 0, 0, w, h

                glyph_bboxes.append((int(l), int(t), int(r), int(b)))

                # advance
                try:
                    adv_base = float(self.get_glyph_advance(font, ch))
                except Exception:
                    adv_base = float(r - l) if (r > l) else float(base_size)

                adv = adv_base * (1.0 + char_gap_rel)
                adv += float(char_gap_px)
                adv = max(1, int(round(adv)))
                advances.append(adv)

            if glyph_bboxes:
                min_left  = min(bb[0] for bb in glyph_bboxes)
                max_right = max(bb[2] for bb in glyph_bboxes)
                left_comp = max(0, -min_left)
                line_w = int(sum(advances) + left_comp)
            else:
                line_w = 0

            line_h = safe_line_h

            line_advances.append(advances)
            line_glyph_bboxes.append(glyph_bboxes)
            line_widths.append(line_w)
            line_heights.append(line_h)

        # 4) Общий размер канвы
        total_w = max(1, (max(line_widths) if len(line_widths) else 1)) + 2 * int(fp_pad_px)
        total_h = 2 * int(fp_pad_px)
        for i, lh in enumerate(line_heights):
            total_h += lh
            if i + 1 < len(line_heights):
                total_h += int(line_gap_px)
        total_h = max(1, total_h)

        canvas = Image.new("L", (total_w, total_h), 0)
        draw   = ImageDraw.Draw(canvas)

        # 5) Рендер и сбор bb_char_xywh
        bb_list = []
        y_cursor = int(fp_pad_px)

        for line_idx, (chars, wbflags, advances, glyph_bboxes) in enumerate(
            zip(per_line_chars, per_line_wbflags, line_advances, line_glyph_bboxes)
        ):
            x_cursor = int(fp_pad_px)

            if glyph_bboxes:
                min_left  = min(bb[0] for bb in glyph_bboxes)
                min_top   = min(bb[1] for bb in glyph_bboxes)
            else:
                min_left, min_top = 0, 0

            base_x_shift = max(0, -min_left)
            base_y_shift = max(0, -min_top)

            line_h = line_heights[line_idx]

            for (ch, adv, (l, t, r, b)) in zip(chars, advances, glyph_bboxes):
                gw, gh = (r - l), (b - t)

                draw_x = x_cursor + (l + base_x_shift)
                draw_y = y_cursor + base_y_shift

                # один раз рисуем глиф
                draw.text((draw_x, draw_y), ch, fill=255, font=pil_font)

                bb_list.append([int(draw_x), int(y_cursor), int(gw), int(line_h)])

                x_cursor += adv

            y_cursor += line_h
            if line_idx + 1 < len(line_heights):
                y_cursor += int(line_gap_px)

        txt_arr = np.array(canvas, dtype=np.uint8)

        # 6) Чуть поправим верх/низ (как было)
        if txt_arr.size:
            nonzero_rows = np.where(np.any(txt_arr > 0, axis=1))[0]
            if nonzero_rows.size:
                top    = int(nonzero_rows[0])
                bottom = int(nonzero_rows[-1])
                H      = txt_arr.shape[0]

                margin_target = max(4, int(0.15 * base_size))

                curr_top_margin    = top
                curr_bottom_margin = H - 1 - bottom

                pad_top    = max(0, margin_target - curr_top_margin)
                pad_bottom = max(0, margin_target - curr_bottom_margin)

                if pad_top > 0 or pad_bottom > 0:
                    txt_arr = np.pad(
                        txt_arr,
                        ((pad_top, pad_bottom), (0, 0)),
                        mode="constant"
                    )
                    for i in range(len(bb_list)):
                        bb_list[i][1] += pad_top

        # 7) Жёсткая бинаризация + более жирный штрих + «ореол» без прозрачности
        if txt_arr.size:
            import cv2

            # 7.1. Убираем полутон: только 0 или 255
            _, txt_bin = cv2.threshold(txt_arr, 0, 255, cv2.THRESH_BINARY)

            # 7.2. Основное «жирное» ядро текста (2 итерации дилатации)
            core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            txt_core = cv2.dilate(txt_bin, core_kernel, iterations=2)

            # 7.3. Дополнительный "ореол" вокруг текста
            glow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            txt_glow = cv2.dilate(txt_bin, glow_kernel, iterations=1)
            _, txt_glow = cv2.threshold(txt_glow, 0, 255, cv2.THRESH_BINARY)

            # 7.4. Итоговая маска: ядро + ореол, всё бинарно
            txt_arr = np.maximum(txt_core, txt_glow)

        txt_str = "\n".join(lines)
        bb_char_xywh = (
            np.array(bb_list, dtype=np.int32)
            if len(bb_list)
            else np.zeros((0, 4), dtype=np.int32)
        )

        return txt_arr, txt_str, bb_char_xywh



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
        Samples from the font state distribution
        """
        return {
            'font': self.fonts[int(np.random.randint(0, len(self.fonts)))],
            'size': self.size[1]*np.random.randn() + self.size[0],
            'underline': np.random.rand() < self.underline,
            'underline_adjustment': max(2.0, min(-2.0, self.underline_adjustment[1]*np.random.randn() + self.underline_adjustment[0])),
            'strong': np.random.rand() < self.strong,
            'oblique': np.random.rand() < self.oblique,
            'strength': (self.strength[1] - self.strength[0])*np.random.rand() + self.strength[0],
            'char_spacing': int(self.kerning[3]*(np.random.beta(self.kerning[0], self.kerning[1])) + self.kerning[2]),
            'border': np.random.rand() < self.border,
            'random_caps': np.random.rand() < self.random_caps,
            'capsmode': random.choice(self.capsmode),
            'curved': np.random.rand() < self.curved,
            'random_kerning': np.random.rand() < self.random_kerning,
            'random_kerning_amount': self.random_kerning_amount,
        }

    def init_font(self,fs):
        """
        Initializes a pygame font.
        FS : font-state sample
        """
        font = freetype.Font(fs['font'], size=fs['size'])
        font.underline = fs['underline']
        font.underline_adjustment = fs['underline_adjustment']
        font.strong = fs['strong']
        font.oblique = fs['oblique']
        font.strength = fs['strength']
        char_spacing = fs['char_spacing']
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