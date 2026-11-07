"""OCR со скриншота.

Tesseract хорошо работает только на чёрном тексте по белому фону с высотой
символов ~25-40px. Скриншоты обычно другие: мелкий текст, тёмные темы,
цветные/градиентные фоны, кнопки с инвертированным текстом, декоративные
шрифты. Поэтому изображение готовится в нескольких вариантах, каждый
распознаётся параллельно, а в ответ идёт вариант с лучшей оценкой по
уверенности Tesseract.
"""

import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

LANG_SETS = ("eng+rus", "jpn+eng")
TARGET_CHAR_HEIGHT = 34  # высота символа (px), на которой Tesseract точнее всего
PAD = 20
MAX_PIXELS = 16_000_000  # предел площади после масштабирования

CJK = r"぀-ヿ㐀-鿿＀-￯"


def fixJapaneseSpaces(text):
    return re.sub(rf"(?<=[{CJK}])\s+(?=[{CJK}])", "", text)


# ---------------------------------------------------------------- подготовка


def load_color(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"не удалось открыть изображение: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # прозрачные пиксели -> белый фон
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return img


def stroke_map(gray, k):
    """Штрихи текста независимо от полярности и плавного фона."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)  # тёмное на светлом
    light = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)  # светлое на тёмном
    return cv2.max(dark, light)


def estimate_char_height(gray):
    """Медианная высота символов по связным компонентам."""
    k = max(9, min(gray.shape) // 8) | 1
    k = min(k, 41)
    strokes = stroke_map(gray, k)
    _, bw = cv2.threshold(strokes, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return None
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    w = stats[1:, cv2.CC_STAT_WIDTH]
    area = stats[1:, cv2.CC_STAT_AREA]
    ok = (h >= 4) & (h < gray.shape[0] * 0.9) & (area >= 6) & (w < h * 6)
    if ok.sum() < 3:
        return None
    # верхний квартиль: игнорируем точки, запятые и обрывки штрихов
    return float(np.percentile(h[ok], 75))


def choose_scale(gray):
    ch = estimate_char_height(gray)
    if ch is None:
        # не удалось оценить — для типичного скриншота мелкий текст
        scale = 3.0 if max(gray.shape) < 1500 else 2.0
    else:
        scale = float(np.clip(TARGET_CHAR_HEIGHT / ch, 0.5, 5.0))
    # не раздуваем огромные скриншоты (память и время Tesseract)
    max_scale = (MAX_PIXELS / (gray.shape[0] * gray.shape[1])) ** 0.5
    return min(scale, max(max_scale, 0.5))


def resize(img, scale):
    if abs(scale - 1.0) < 0.05:
        return img
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)


def normalize_polarity(bw):
    """Текст — чёрный, фон — белый (текст всегда в меньшинстве)."""
    border = np.concatenate([bw[0], bw[-1], bw[:, 0], bw[:, -1]])
    bg_white = border.mean() > 127 if border.size else bw.mean() > 127
    return bw if bg_white else cv2.bitwise_not(bw)


def pad(img):
    return cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)


def best_contrast_channel(color):
    """Канал с наибольшим контрастом — для цветного текста на цветном фоне,
    который в оттенках серого сливается с фоном."""
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    channels = [lab[:, :, 0], lab[:, :, 1], lab[:, :, 2], hsv[:, :, 1],
                color[:, :, 0], color[:, :, 1], color[:, :, 2]]
    best, best_score = channels[0], -1.0
    for full in channels:
        full = cv2.normalize(full, None, 0, 255, cv2.NORM_MINMAX)
        ch = full[::4, ::4]  # для оценки хватит прореженной копии
        t, _ = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lo, hi = ch[ch <= t], ch[ch > t]
        if lo.size == 0 or hi.size == 0:
            continue
        # межклассовая разница, нормированная на внутриклассовый разброс
        score = (hi.mean() - lo.mean()) / (lo.std() + hi.std() + 1.0)
        if score > best_score:
            best, best_score = full, score
    return best


def textness(mask, char_h=TARGET_CHAR_HEIGHT):
    """Насколько маска похожа на строки текста: много компонент с высотой
    около ожидаемой, без огромных сплошных пятен."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 0.0
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    w = stats[1:, cv2.CC_STAT_WIDTH]
    area = stats[1:, cv2.CC_STAT_AREA]
    fill = area / np.maximum(h * w, 1)
    glyph = ((h > char_h * 0.4) & (h < char_h * 2.2)
             & (w < char_h * 3) & (fill > 0.1) & (fill < 0.85))
    if glyph.sum() < 3:
        return 0.0
    junk = area[~glyph].sum() / max(area.sum(), 1)
    return glyph.sum() * (1.0 - junk)


def drop_non_glyphs(mask, char_h=TARGET_CHAR_HEIGHT):
    """Убирает из маски крупные пятна фона, оставляя буквы и мелкие знаки."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h = stats[:, cv2.CC_STAT_HEIGHT]
    w = stats[:, cv2.CC_STAT_WIDTH]
    keep = (h < char_h * 2.5) & (w < char_h * 4)
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def color_mask_rules(lab, centers):
    """Правила-маски: принадлежность к кластеру цвета + крайние цвета."""
    # ближайший центр через таблицу по квантованному цвету (32 уровня на канал)
    q = (np.arange(32, dtype=np.float32) * 8 + 4)
    grid = np.stack(np.meshgrid(q, q, q, indexing="ij"), -1).reshape(-1, 3)
    lut = ((grid[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)
    idx = ((lab[:, :, 0].astype(np.int32) >> 3) << 10
           | (lab[:, :, 1].astype(np.int32) >> 3) << 5
           | (lab[:, :, 2].astype(np.int32) >> 3))
    labels = lut.astype(np.uint8)[idx]
    L = lab[:, :, 0]
    chroma = (np.abs(lab[:, :, 1].astype(np.int16) - 128)
              + np.abs(lab[:, :, 2].astype(np.int16) - 128))
    masks = [labels == i for i in range(len(centers))]
    masks += [(L > 225) & (chroma < 20), L < 40]
    return masks


def clean_mask(m, char_h):
    m = np.where(m, 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return drop_non_glyphs(m, char_h)


def color_masks(color, k=6, top=2):
    """Текст обычно одного цвета. Строим маски по кластерам цветов и по
    крайним цветам (почти белый / почти чёрный) и берём те, что больше
    всего похожи на буквы (пёстрый фон, текст поверх картинок).
    Выбор делается на уменьшенной копии, итоговые маски — в полном размере."""
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
    if lab.shape[0] * lab.shape[1] < k * 10:
        return []
    flat = lab.reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(0)
    samples = flat[rng.choice(len(flat), min(len(flat), 20000), replace=False)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(samples, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)

    small = cv2.resize(lab, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    char_small = TARGET_CHAR_HEIGHT / 2
    scored = []
    for i, m in enumerate(color_mask_rules(small, centers)):
        s = textness(clean_mask(m, char_small), char_small)
        if s > 0:
            scored.append((s, i))
    scored.sort(reverse=True)
    if not scored:
        return []
    full = color_mask_rules(lab, centers)
    return [cv2.bitwise_not(clean_mask(full[i], TARGET_CHAR_HEIGHT))
            for _, i in scored[:top]]


def make_variants(color):
    gray0 = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    scale = choose_scale(gray0)
    color = resize(color, scale)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    variants = {}

    # 1. Просто серый, приведённый к тёмному тексту на светлом
    otsu_t, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    g = gray if normalize_polarity(bw) is bw else cv2.bitwise_not(gray)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    variants["gray"] = g

    # 2. Глобальный порог Оцу
    variants["otsu"] = normalize_polarity(bw)

    # 3. Выравнивание фона + адаптивный порог (градиенты, неровная подсветка)
    k = int(TARGET_CHAR_HEIGHT * 2.5) | 1
    bg = cv2.medianBlur(g, min(k, 255) if min(k, 255) % 2 else min(k, 255) - 1)
    flat = cv2.divide(g, bg, scale=255)
    block = int(TARGET_CHAR_HEIGHT * 1.5) | 1
    variants["adaptive"] = cv2.adaptiveThreshold(
        flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 15
    )

    # 4. Top-hat + black-hat: текст любой полярности на сложном фоне
    #    (смешанные светлые/тёмные элементы интерфейса, картинки)
    strokes = stroke_map(gray, int(TARGET_CHAR_HEIGHT * 1.2) | 1)
    _, sb = cv2.threshold(strokes, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sb = cv2.morphologyEx(sb, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    variants["strokes"] = cv2.bitwise_not(sb)

    # 5. Лучший цветовой канал (цветной текст на цветном фоне)
    ch = best_contrast_channel(color)
    _, cb = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants["channel"] = normalize_polarity(cb)

    # 6. Кластер цвета текста (пёстрый фон, текст поверх картинок)
    for i, m in enumerate(color_masks(color)):
        variants[f"color{i}"] = m

    return {name: pad(img) for name, img in variants.items()}


# ---------------------------------------------------------------- tesseract


def run_tesseract(path, lang, psm):
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    out = subprocess.run(
        ["tesseract", path, "stdout", "-l", lang, "--psm", str(psm),
         "-c", "preserve_interword_spaces=1", "tsv"],
        capture_output=True, text=True, env=env,
    ).stdout
    words = []
    for row in out.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12 or cols[0] != "5":
            continue
        text = cols[11].strip()
        if not text:
            continue
        words.append({
            "key": (int(cols[2]), int(cols[3]), int(cols[4])),  # block, par, line
            "left": int(cols[6]),
            "top": int(cols[7]),
            "height": int(cols[9]),
            "conf": float(cols[10]),
            "text": text,
        })
    return words


def is_junk(w):
    t = w["text"]
    has_alnum = any(c.isalnum() for c in t)
    if w["conf"] < 15:
        return True
    if not has_alnum and w["conf"] < 60:
        return True
    return len(t) <= 2 and w["conf"] < 35


def score(words):
    """Сумма уверенности по символам: поощряет и точность, и полноту,
    штрафует мусор с низкой уверенностью."""
    s = 0.0
    for w in words:
        n = sum(c.isalnum() for c in w["text"])
        c = w["conf"] / 100.0
        s += n * (c - 0.5) * 2 if c >= 0 else 0
    return s


def words_to_text(words):
    lines, order = {}, []
    for w in words:
        if is_junk(w):
            continue
        if w["key"] not in lines:
            lines[w["key"]] = []
            order.append(w["key"])
        lines[w["key"]].append(w)
    out, prev = [], None
    for key in order:
        ws = sorted(lines[key], key=lambda w: w["left"])
        top = min(w["top"] for w in ws)
        bottom = max(w["top"] + w["height"] for w in ws)
        if prev is not None:
            p_top, p_bottom, p_key = prev
            line_h = max(p_bottom - p_top, bottom - top, 1)
            new_par = key[:2] != p_key[:2]
            # пустая строка — только при заметном отступе между абзацами
            if new_par and (top - p_bottom > line_h * 0.8 or top < p_top):
                out.append("")
        out.append(" ".join(w["text"] for w in ws))
        prev = (top, bottom, key)
    return "\n".join(out).strip()


def getText(path):
    color = load_color(path)
    variants = make_variants(color)

    with tempfile.TemporaryDirectory(prefix="screentext-") as tmp:
        files = {}
        for name, img in variants.items():
            files[name] = os.path.join(tmp, f"{name}.png")
            cv2.imwrite(files[name], img)

        jobs = [(name, lang, 3) for name in files for lang in LANG_SETS]
        # PSM 11 (разреженный текст) — для интерфейсов с разбросанными надписями
        jobs += [(name, LANG_SETS[0], 11) for name in ("gray", "strokes")]

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            results = list(pool.map(
                lambda j: (j, run_tesseract(files[j[0]], j[1], j[2])), jobs
            ))

    best_job, best_words = max(results, key=lambda r: score(r[1]))
    if os.environ.get("SCREENTEXT_DEBUG"):
        for job, words in sorted(results, key=lambda r: -score(r[1])):
            print(f"{score(words):8.1f}  {job}")

    return fixJapaneseSpaces(words_to_text(best_words))
