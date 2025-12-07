import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.interpolate as si
import scipy.ndimage as scim 
import scipy.ndimage.interpolation as sii
import os
import os.path as osp
#import cPickle as cp
import _pickle as cp
#import Image
from PIL import Image
from poisson_reconstruct import blit_images
import pickle

def sample_weighted(p_dict):
    keys = list(p_dict.keys())
    probs = np.array(list(p_dict.values()), dtype=float)
    probs = probs / probs.sum()
    return np.random.choice(keys, p=probs)

class Layer(object):

    def __init__(self,alpha,color):

        # alpha for the whole image:
        assert alpha.ndim==2
        self.alpha = alpha
        [n,m] = alpha.shape[:2]

        color=np.atleast_1d(np.array(color)).astype('uint8')
        # color for the image:
        if color.ndim==1: # constant color for whole layer
            ncol = color.size
            if ncol == 1 : #grayscale layer
                self.color = color * np.ones((n,m,3),'uint8')
            if ncol == 3 : 
                self.color = np.ones((n,m,3),'uint8') * color[None,None,:]
        elif color.ndim==2: # grayscale image
            self.color = np.repeat(color[:,:,None],repeats=3,axis=2).copy().astype('uint8')
        elif color.ndim==3: #rgb image
            self.color = color.copy().astype('uint8')
        else:
            print (color.shape)
            raise Exception("color datatype not understood")

class FontColor(object):

    def __init__(self, col_file):
        with open(col_file,'rb') as f:
            #self.colorsRGB = cp.load(f)
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load()
            self.colorsRGB = p
        self.ncol = self.colorsRGB.shape[0]

        # convert color-means from RGB to LAB for better nearest neighbour
        # computations:
        self.colorsLAB = np.r_[self.colorsRGB[:,0:3], self.colorsRGB[:,6:9]].astype('uint8')
        self.colorsLAB = np.squeeze(cv.cvtColor(self.colorsLAB[None,:,:],cv.COLOR_RGB2Lab))


    def sample_normal(self, col_mean, col_std):
        """
        sample from a normal distribution centered around COL_MEAN 
        with standard deviation = COL_STD.
        """
        col_sample = col_mean + col_std * np.random.randn()
        return np.clip(col_sample, 0, 255).astype('uint8')

    def sample_from_data(self, bg_mat):
        """
        bg_mat : this is a nxmx3 RGB image.
        
        returns a tuple : (RGB_foreground, RGB_background)
        each of these is a 3-vector.
        """
        bg_orig = bg_mat.copy()
        bg_mat = cv.cvtColor(bg_mat, cv.COLOR_RGB2Lab)
        bg_mat = np.reshape(bg_mat, (np.prod(bg_mat.shape[:2]),3))
        bg_mean = np.mean(bg_mat,axis=0)

        norms = np.linalg.norm(self.colorsLAB-bg_mean[None,:], axis=1)
        # choose a random color amongst the top 3 closest matches:
        #nn = np.random.choice(np.argsort(norms)[:3]) 
        nn = np.argmin(norms)

        ## nearest neighbour color:
        data_col = self.colorsRGB[np.mod(nn,self.ncol),:]

        col1 = self.sample_normal(data_col[:3],data_col[3:6])
        col2 = self.sample_normal(data_col[6:9],data_col[9:12])

        if nn < self.ncol:
            return (col2, col1)
        else:
            # need to swap to make the second color close to the input backgroun color
            return (col1, col2)

    def mean_color(self, arr):
        col = cv.cvtColor(arr, cv.COLOR_RGB2HSV)
        col = np.reshape(col, (np.prod(col.shape[:2]),3))
        col = np.mean(col,axis=0).astype('uint8')
        return np.squeeze(cv.cvtColor(col[None,None,:],cv.COLOR_HSV2RGB))

    def invert(self, rgb):
        rgb = 127 + rgb
        return rgb

    def complement(self, rgb_color):
        """
        return a color which is complementary to the RGB_COLOR.
        """
        col_hsv = np.squeeze(cv.cvtColor(rgb_color[None,None,:], cv.COLOR_RGB2HSV))
        col_hsv[0] = col_hsv[0] + 128 #uint8 mods to 255
        col_comp = np.squeeze(cv.cvtColor(col_hsv[None,None,:],cv.COLOR_HSV2RGB))
        return col_comp

    def triangle_color(self, col1, col2):
        """
        Returns a color which is "opposite" to both col1 and col2.
        """
        col1, col2 = np.array(col1), np.array(col2)
        col1 = np.squeeze(cv.cvtColor(col1[None,None,:], cv.COLOR_RGB2HSV))
        col2 = np.squeeze(cv.cvtColor(col2[None,None,:], cv.COLOR_RGB2HSV))
        h1, h2 = col1[0], col2[0]
        if h2 < h1 : h1,h2 = h2,h1 #swap
        dh = h2-h1
        if dh < 127: dh = 255-dh
        col1[0] = h1 + dh/2
        return np.squeeze(cv.cvtColor(col1[None,None,:],cv.COLOR_HSV2RGB))

    def change_value(self, col_rgb, v_std=50):
        col = np.squeeze(cv.cvtColor(col_rgb[None,None,:], cv.COLOR_RGB2HSV))
        x = col[2]
        vs = np.linspace(0,1)
        ps = np.abs(vs - x/255.0)
        ps /= np.sum(ps)
        v_rand = np.clip(np.random.choice(vs,p=ps) + 0.1*np.random.randn(),0,1)
        col[2] = 255*v_rand
        return np.squeeze(cv.cvtColor(col[None,None,:],cv.COLOR_HSV2RGB))


class Colorize(object):

    def __init__(self, model_dir='data'):#, im_path):
        # # get a list of background-images:
        # imlist = [osp.join(im_path,f) for f in os.listdir(im_path)]
        # self.bg_list = [p for p in imlist if osp.isfile(p)]

        self.font_color = FontColor(col_file=osp.join(model_dir,'models/colors_new.cp'))

        # probabilities of different text-effects:
        self.p_bevel = 0.05 # add bevel effect to text
        self.p_outline = 0.05 # just keep the outline of the text
        self.p_drop_shadow = 0.15
        self.p_border = 0.15
        self.p_displacement = 0.30 # add background-based bump-mapping
        self.p_texture = 0.0 # use an image for coloring text


    def _sample_text_color(self):
        """
        Возвращает (R, G, B). Добавлен фильтр, запрещающий слишком тёмные цвета.
        """
        import numpy as np

        for _ in range(50):  # несколько попыток, чтобы найти не слишком тёмный цвет
            c = np.random.randint(0, 256, size=3)
            if np.mean(c) > 60:  # яркость > 60 → не чёрный
                return tuple(int(x) for x in c)
        return (80, 80, 80)  # fallback — серый


    def drop_shadow(self, alpha, theta, shift, size, op=0.80):
        """
        alpha : alpha layer whose shadow need to be cast
        theta : [0,2pi] -- the shadow direction
        shift : shift in pixels of the shadow
        size  : size of the GaussianBlur filter
        op    : opacity of the shadow (multiplying factor)

        @return : alpha of the shadow layer
                  (it is assumed that the color is black/white)
        """
        if size%2==0:
            size -= 1
            size = max(1,size)
        shadow = cv.GaussianBlur(alpha,(size,size),0)
        [dx,dy] = shift * np.array([-np.sin(theta), np.cos(theta)])
        shadow = op*sii.shift(shadow, shift=[dx,dy],mode='constant',cval=0)
        return shadow.astype('uint8')

    def border(self, alpha, size, kernel_type='RECT'):
        """
        alpha : alpha layer of the text
        size  : size of the kernel
        kernel_type : one of [rect,ellipse,cross]

        @return : alpha layer of the border (color to be added externally).
        """
        kdict = {'RECT':cv.MORPH_RECT, 'ELLIPSE':cv.MORPH_ELLIPSE,
                 'CROSS':cv.MORPH_CROSS}
        kernel = cv.getStructuringElement(kdict[kernel_type],(size,size))
        border = cv.dilate(alpha,kernel,iterations=1) # - alpha
        return border

    def blend(self,cf,cb,mode='normal'):
        return cf

    def merge_two(self,fore,back,blend_type=None):
        """
        merge two FOREground and BACKground layers.
        ref: https://en.wikipedia.org/wiki/Alpha_compositing
        ref: Chapter 7 (pg. 440 and pg. 444):
             http://partners.adobe.com/public/developer/en/pdf/PDFReference.pdf
        """
        a_f = fore.alpha/255.0
        a_b = back.alpha/255.0
        c_f = fore.color
        c_b = back.color

        a_r = a_f + a_b - a_f*a_b
        if blend_type != None:
            c_blend = self.blend(c_f, c_b, blend_type)
            c_r = (   ((1-a_f)*a_b)[:,:,None] * c_b
                    + ((1-a_b)*a_f)[:,:,None] * c_f
                    + (a_f*a_b)[:,:,None] * c_blend   )
        else:
            c_r = (   ((1-a_f)*a_b)[:,:,None] * c_b
                    + a_f[:,:,None]*c_f    )

        return Layer((255*a_r).astype('uint8'), c_r.astype('uint8'))

    def merge_down(self, layers, blends=None):
        """
        layers  : [l1,l2,...ln] : a list of LAYER objects.
                 l1 is on the top, ln is the bottom-most layer.
        blend   : the type of blend to use. Should be n-1.
                 use None for plain alpha blending.
        Note    : (1) it assumes that all the layers are of the SAME SIZE.
        @return : a single LAYER type object representing the merged-down image
        """
        nlayers = len(layers)
        if nlayers > 1:
            [n,m] = layers[0].alpha.shape[:2]
            out_layer = layers[-1]
            for i in range(-2,-nlayers-1,-1):
                blend=None
                if blends is not None:
                    blend = blends[i+1]
                    out_layer = self.merge_two(fore=layers[i], back=out_layer,blend_type=blend)
            return out_layer
        else:
            return layers[0]

    def resize_im(self, im, osize):
        return np.array(Image.fromarray(im).resize(osize[::-1], Image.NEAREST))
        
    def occlude(self):
        """
        somehow add occlusion to text.
        """
        pass

    def color_border(self, col_text, col_bg):
        """
        Decide on a color for the border:
            - could be the same as text-color but lower/higher 'VALUE' component.
            - could be the same as bg-color but lower/higher 'VALUE'.
            - could be 'mid-way' color b/w text & bg colors.
        """
        choice = np.random.choice(3)

        col_text = cv.cvtColor(col_text, cv.COLOR_RGB2HSV)
        col_text = np.reshape(col_text, (np.prod(col_text.shape[:2]),3))
        col_text = np.mean(col_text,axis=0).astype('uint8')

        vs = np.linspace(0,1)
        def get_sample(x):
            ps = np.abs(vs - x/255.0)
            ps /= np.sum(ps)
            v_rand = np.clip(np.random.choice(vs,p=ps) + 0.1*np.random.randn(),0,1)
            return 255*v_rand

        # first choose a color, then inc/dec its VALUE:
        if choice==0:
            # increase/decrease saturation:
            col_text[0] = get_sample(col_text[0]) # saturation
            col_text = np.squeeze(cv.cvtColor(col_text[None,None,:],cv.COLOR_HSV2RGB))
        elif choice==1:
            # get the complementary color to text:
            col_text = np.squeeze(cv.cvtColor(col_text[None,None,:],cv.COLOR_HSV2RGB))
            col_text = self.font_color.complement(col_text)
        else:
            # choose a mid-way color:
            col_bg = cv.cvtColor(col_bg, cv.COLOR_RGB2HSV)
            col_bg = np.reshape(col_bg, (np.prod(col_bg.shape[:2]),3))
            col_bg = np.mean(col_bg,axis=0).astype('uint8')
            col_bg = np.squeeze(cv.cvtColor(col_bg[None,None,:],cv.COLOR_HSV2RGB))
            col_text = np.squeeze(cv.cvtColor(col_text[None,None,:],cv.COLOR_HSV2RGB))
            col_text = self.font_color.triangle_color(col_text,col_bg)

        # now change the VALUE channel:        
        col_text = np.squeeze(cv.cvtColor(col_text[None,None,:],cv.COLOR_RGB2HSV))
        col_text[2] = get_sample(col_text[2]) # value
        return np.squeeze(cv.cvtColor(col_text[None,None,:],cv.COLOR_HSV2RGB))

    def color_text(self, text_arr, h, bg_arr):
        """
        Выбор цвета текста с защитой от слишком тёмных вариантов
        и с проверкой контраста к локальному фону в месте, где будет текст.

        text_arr : (H, W) uint8 маска текста (0/255 или с полутоном)
        h        : минимальная высота символа (px) — тут не используется
        bg_arr   : (H, W, 3) uint8 фоновое RGB изображение

        return: Layer(alpha=text_arr, color=fg_col), fg_col, bg_col
        """
        # 1) Базовый выбор по твоей логике
        fg_col, bg_col = self.font_color.sample_from_data(bg_arr)

        # --- Вспомогательные функции ---
        def _to_uint8_tuple(x):
            x = np.asarray(x, dtype=np.int32)
            return (int(np.clip(x[0], 0, 255)),
                    int(np.clip(x[1], 0, 255)),
                    int(np.clip(x[2], 0, 255)))

        def _rel_luminance(rgb):
            # относительная яркость (приближённо как в WCAG)
            r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
            return 0.2126*r + 0.7152*g + 0.0722*b

        # 2) Собираем локальный фон там, где будет текст
        mask_bool = text_arr > 0
        if np.any(mask_bool):
            local_bg = bg_arr[mask_bool]
            local_bg_mean = np.mean(local_bg, axis=0).astype(np.uint8)
        else:
            # если маска пустая (теоретически) — берём средний цвет всего фона
            local_bg_mean = np.mean(np.mean(bg_arr, axis=0), axis=0).astype(np.uint8)

        # 3) Анти-тёмный фильтр: поднимем яркость, если слишком темно
        if np.mean(fg_col) < 60:
            # мягко подтягиваем к светло-серому
            fg_col = _to_uint8_tuple(0.5*np.array(fg_col) + 0.5*np.array([128, 128, 128]))
        else:
            fg_col = _to_uint8_tuple(fg_col)

        # 4) Проверка контраста с локальным фоном
        def _lum(rgb_u8):
            r, g, b = rgb_u8[0]/255.0, rgb_u8[1]/255.0, rgb_u8[2]/255.0
            return 0.2126*r + 0.7152*g + 0.0722*b

        L_fg = _lum(fg_col)
        L_bg = _lum(local_bg_mean)

        # адаптивный порог: на светлом фоне требуем больше
        min_contrast = 0.20 + 0.20 * L_bg  # 0.20..0.40

        def _ensure_contrast(fg_rgb, bg_rgb, min_c):
            Lf, Lb = _lum(fg_rgb), _lum(bg_rgb)
            if abs(Lf - Lb) >= min_c:
                return _to_uint8_tuple(fg_rgb)

            # 1) если фон светлый — затемняем; если тёмный — осветляем
            if Lb > 0.5:
                cand = 0.35 * np.array(fg_rgb)
            else:
                cand = 0.6 * np.array(fg_rgb) + 0.4 * np.array([255, 255, 255])
            if abs(_lum(cand) - Lb) >= min_c:
                return _to_uint8_tuple(cand)

            # 2) инверсия от локального фона
            cand = 255 - np.array(local_bg_mean, dtype=np.float32)
            if abs(_lum(cand) - Lb) >= min_c:
                return _to_uint8_tuple(cand)

            # 3) крайний случай — сдвиг Value в HSV
            hsv = cv.cvtColor(np.uint8([[fg_rgb]]), cv.COLOR_RGB2HSV).astype(np.float32)
            if Lb > 0.5:
                hsv[0,0,2] = max(0, hsv[0,0,2] - 80)  # затемнить
            else:
                hsv[0,0,2] = min(255, hsv[0,0,2] + 80)  # осветлить
            cand = cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2RGB)[0,0,:]
            return _to_uint8_tuple(cand)

        fg_col = _ensure_contrast(fg_col, local_bg_mean, min_contrast)

        return Layer(alpha=text_arr, color=np.array(fg_col, dtype=np.uint8)), fg_col, bg_col



    def process(self, text_arr, bg_arr, min_h):
        """
        Жёсткое наложение текста без прозрачности/блюра/поассона.
        text_arr : (h x w) uint8 маска (может быть с серыми краями)
        bg_arr   : (h x w x 3) uint8 фон
        min_h    : не используется тут, оставлен для сигнатуры
        """
        # 1) Жёсткая бинаризация маски (убираем антиалиасинг краёв)
        # Если у тебя маски в 0/255, можно поднять порог до 127.
        _, mask = cv.threshold(text_arr, 127, 255, cv.THRESH_BINARY)

        # 2) Выбираем цвет текста (оставим твою логику подбора по фону)
        l_text, fg_col, bg_col = self.color_text(mask, min_h, bg_arr)

        # 3) Строим однотонный “слой” текста
        text_rgb = np.zeros_like(bg_arr, dtype=np.uint8)
        text_rgb[mask > 0] = fg_col  # закрасить только пиксели текста

        # 4) Жёсткая композиция без альфа-смешивания
        out = bg_arr.copy()
        out[mask > 0] = np.array(fg_col, dtype=np.uint8)
        return out


    def check_perceptible(self, txt_mask, bg, txt_bg):
        """
        --- DEPRECATED; USE GRADIENT CHECKING IN POISSON-RECONSTRUCT INSTEAD ---

        checks if the text after merging with background
        is still visible.
        txt_mask (hxw) : binary image of text -- 255 where text is present
                                                   0 elsewhere
        bg (hxwx3) : original background image WITHOUT any text.
        txt_bg (hxwx3) : image with text.
        """
        bgo,txto = bg.copy(), txt_bg.copy()
        txt_mask = txt_mask.astype('bool')
        bg = cv.cvtColor(bg.copy(), cv.COLOR_RGB2Lab)
        txt_bg = cv.cvtColor(txt_bg.copy(), cv.COLOR_RGB2Lab)
        bg_px = bg[txt_mask,:]
        txt_px = txt_bg[txt_mask,:]
        bg_px[:,0] *= 100.0/255.0 #rescale - L channel
        txt_px[:,0] *= 100.0/255.0

        diff = np.linalg.norm(bg_px-txt_px,ord=None,axis=1)
        diff = np.percentile(diff,[10,30,50,70,90])
        print ("color diff percentile :", diff)
        return diff, (bgo,txto)

    def color(self, bg_arr, text_arr, hs, place_order=None, pad=20):
        """
        Жёсткое наложение текста поверх bg_arr с коррекцией контраста.
        text_arr : list[np.ndarray HxW uint8/bool] — бинарные/полутоновые маски текста (0/255 или 0/1)
        hs       : list/np.ndarray/scalar — минимальная высота символа(ов) для каждой маски
        place_order : порядок отрисовки (по умолчанию — как есть)
        pad      : безопасный паддинг вокруг bbox (обрезается по границам канвы)
        return   : bg_arr с прорисованным текстом
        """
        import numpy as np
        import cv2 as cv

        # 0) копия и гарантия 3-канального RGB
        out = bg_arr.copy()
        if out.ndim == 2 or out.shape[2] == 1:
            out = np.repeat(out[:, :, None], 3, axis=2)

        H, W = out.shape[:2]

        # 1) нормализуем order
        if place_order is None:
            place_order = np.arange(len(text_arr))
        else:
            place_order = np.asarray(place_order, dtype=int)

        # 2) приведём hs к индексации (если это скаляр — использовать его для всех)
        def _min_h_at(i):
            if isinstance(hs, (list, tuple, np.ndarray)):
                return float(hs[i])
            return float(hs)

        # 3) основной цикл (верхние маски рисуем последними — как в исходнике)
        for i in place_order[::-1]:
            mask_full = text_arr[i]
            if mask_full is None:
                continue

            # допускаем bool, 0/1 или 0..255
            mask_full = np.asarray(mask_full)
            if mask_full.dtype != np.uint8:
                mask_bin = (mask_full > 0).astype(np.uint8)
            else:
                mask_bin = (mask_full > 0).astype(np.uint8)

            if not np.any(mask_bin):
                continue  # пустая маска — пропуск

            # 3.1) bbox по маске (глобальные координаты на канве)
            ys, xs = np.where(mask_bin > 0)
            y0, x0 = int(ys.min()), int(xs.min())
            y1, x1 = int(ys.max()), int(xs.max())
            h = int(y1 - y0 + 1)
            w = int(x1 - x0 + 1)

            # 3.2) плотный вырез маски (по bbox)
            text_patch = mask_bin[y0:y0 + h, x0:x0 + w]

            # 3.3) безопасный паддинг (обрезаем по краям канвы)
            top    = min(int(pad), y0)
            left   = min(int(pad), x0)
            bottom = min(int(pad), max(0, H - (y0 + h)))
            right  = min(int(pad), max(0, W - (x0 + w)))

            if top or left or bottom or right:
                text_patch = np.pad(
                    text_patch,
                    ((top, bottom), (left, right)),
                    mode='constant'
                )
                y0 -= top
                x0 -= left
                h = text_patch.shape[0]
                w = text_patch.shape[1]

            # 3.4) финальная подстраховка границ
            # (на случай, если что-то округлилось и вышло за край)
            if y0 < 0:
                # сдвигаем окно вверх, а паддинг фактически срезаем
                cut = -y0
                text_patch = text_patch[cut:, :]
                y0 = 0
                h = text_patch.shape[0]
            if x0 < 0:
                cut = -x0
                text_patch = text_patch[:, cut:]
                x0 = 0
                w = text_patch.shape[1]

            if y0 + h > H:
                cut = (y0 + h) - H
                if cut > 0:
                    text_patch = text_patch[:-cut, :]
                    h = text_patch.shape[0]
            if x0 + w > W:
                cut = (x0 + w) - W
                if cut > 0:
                    text_patch = text_patch[:, :-cut]
                    w = text_patch.shape[1]

            if h <= 0 or w <= 0:
                continue  # после обрезки ничего не осталось

            bg_patch = out[y0:y0 + h, x0:x0 + w, :]

            # 3.5) рендер патча
            min_h = _min_h_at(i)
            # self.process ожидает uint8 маску (0/255) — обеспечим
            text_patch_u8 = (text_patch.astype(np.uint8) * 255)
            out_patch = self.process(text_patch_u8, bg_patch, min_h)

            # 3.6) записываем назад
            out[y0:y0 + h, x0:x0 + w, :] = out_patch

        return out
