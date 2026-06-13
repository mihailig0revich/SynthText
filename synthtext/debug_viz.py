import cv2
import numpy as np


def init_interactive_matplotlib():
    """Enable interactive matplotlib and warm up the first figure."""
    import matplotlib.pyplot as plt

    plt.ion()
    try:
        import matplotlib
        backend = matplotlib.get_backend()
        print("[VIZ] matplotlib backend:", backend)
        if "agg" in str(backend).lower():
            return plt

        figure = plt.figure(99)
        plt.plot([0, 1], [0, 1])
        plt.show(block=False)
        plt.pause(0.05)
        plt.close(figure)
    except Exception:
        pass
    return plt


def rgb_for_matplotlib(im):
    if im is None:
        return im
    if im.ndim == 3 and im.shape[2] == 3:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def to_rgb(arr):
    if arr is None:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def stable_matplotlib_draw(fig=None, pause=0.25):
    import matplotlib.pyplot as plt

    try:
        figure = fig or plt.gcf()
        figure.canvas.draw_idle()
        figure.canvas.flush_events()
        plt.show(block=False)
        plt.pause(pause)
        return True
    except Exception as exc:
        print("[VIZ] Matplotlib draw failed:", exc)
        return False


def cv2_preview(win_name, img_rgb):
    if img_rgb is None:
        return
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        bgr = img_rgb
    cv2.imshow(win_name, bgr)
    cv2.waitKey(1)


def viz_textbb(fignum, text_im, bb_list, alpha=1.0):
    import matplotlib.pyplot as plt

    plt.figure(fignum)
    plt.clf()
    plt.imshow(rgb_for_matplotlib(text_im))
    height, width = text_im.shape[:2]
    for bbs in bb_list:
        ni = bbs.shape[-1]
        for j in range(ni):
            bb = bbs[:, :, j]
            bb = np.c_[bb, bb[:, 0]]
            plt.plot(bb[0, :], bb[1, :], "r", linewidth=2, alpha=alpha)
    plt.gca().set_xlim([0, width - 1])
    plt.gca().set_ylim([height - 1, 0])
    plt.tight_layout()
    if not stable_matplotlib_draw(plt.gcf(), pause=0.35):
        cv2_preview("SynthText (bb)", text_im)


def viz_masks(fignum, rgb, seg, depth, label):
    import matplotlib.pyplot as plt

    def mean_seg(rgb_arr, seg_arr, label_arr):
        mean_img = np.zeros_like(rgb_arr)
        for idx in np.unique(seg_arr.flat):
            mask = seg_arr == idx
            col = np.mean(rgb_arr[mask, :], axis=0)
            mean_img[mask, :] = col[None, None, :]
        mean_img[seg_arr == 0, :] = 0
        return mean_img

    mean_img = mean_seg(rgb, seg, label)
    img = rgb.copy()
    for idx in label:
        mask = seg == idx
        rgb_rand = (255 * np.random.rand(3)).astype("uint8")
        img[mask] = rgb_rand[None, None, :]

    plt.figure(fignum)
    plt.clf()
    ims = [rgb, mean_img, depth, img]
    for i, im in enumerate(ims):
        plt.subplot(2, 2, i + 1)
        plt.imshow(rgb_for_matplotlib(im))
    plt.tight_layout()
    if not stable_matplotlib_draw(plt.gcf(), pause=0.35):
        cv2_preview("SynthText (masks)", rgb)


def viz_generation_overview(fignum, original_rgb, seg, depth, final_rgb):
    import matplotlib.pyplot as plt

    def colorize_seg(seg_arr):
        seg_i = np.asarray(seg_arr, dtype=np.int64)
        out = np.zeros(seg_i.shape + (3,), dtype=np.uint8)
        labels = np.unique(seg_i)
        for lbl in labels:
            if int(lbl) == 0:
                continue
            rng = np.random.default_rng(int(lbl) & 0xFFFFFFFF)
            out[seg_i == lbl] = rng.integers(40, 256, size=3, dtype=np.uint8)
        return out

    def normalize_depth(depth_arr):
        d = np.asarray(depth_arr, dtype=np.float32)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        valid = d[d > 0]
        if valid.size:
            lo = float(np.percentile(valid, 2))
            hi = float(np.percentile(valid, 98))
            if hi <= lo:
                hi = lo + 1.0
            d = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        return d

    plt.figure(fignum)
    plt.clf()

    panels = [
        ("original", rgb_for_matplotlib(original_rgb)),
        ("segmentation", colorize_seg(seg)),
        ("depth", normalize_depth(depth)),
        ("final", rgb_for_matplotlib(final_rgb)),
    ]

    for i, (title, image) in enumerate(panels, start=1):
        plt.subplot(2, 2, i)
        if title == "depth":
            plt.imshow(image, cmap="magma")
        else:
            plt.imshow(image)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    if not stable_matplotlib_draw(plt.gcf(), pause=0.35):
        cv2_preview("SynthText (overview)", final_rgb)
