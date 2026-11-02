"""
MoodWave
========
A small music app with a title page, a song library, and a soft pastel
"cloud" visualiser that reacts to bass, mid/treble frequencies, beats
and tempo.

Run:
    python3 moodwave.py

Songs:
    * Put audio files (mp3, wav, flac, ogg) in the "music" folder that
      MoodWave creates next to this file, OR drag & drop them onto the
      MoodWave window.
    * Audio files sitting next to this file also show up automatically.

Keys:
    Title page     ENTER = play   L = library   ESC = quit
    Library        UP/DOWN + ENTER, or click a song   ESC = back
    Visualiser     SPACE = pause   ESC = back   F = fullscreen

How this file is organised (top to bottom):
    1. Settings & colours  - the numbers and palette everything else uses
    2. Audio analysis      - "listens" to a song and records what it hears
    3. Library             - finds the songs you've added
    4. Drawing helpers     - fonts, text, background and other small tools
    5. The cloud sphere    - the music-reactive cloud itself
    6. The app             - screens, buttons, mouse/keyboard and the main loop
    7. Start               - where the program begins

Every section opens with a short "In plain English" note and a list of helpful
links (docs, GitHub projects, articles). The full list is also in README.md.
"""

import os
import sys
import math
import time
import shutil
import hashlib
import threading
import subprocess
from pathlib import Path

import numpy as np
import pygame


# ==========================================================================
# SETTINGS & COLOURS
# ==========================================================================

#   This is the app's control panel: where your songs live, how smoothly the
#   animation runs, and which colours are used. Change a value here and the whole
#   app changes with it, so it's the safest place to start experimenting.
#
# Helpful resources:
#   - Python pathlib (how folders and files are handled)
#     https://docs.python.org/3/library/pathlib.html
#   - Smoothstep (the easing curve used to blend between palette colours)
#     https://en.wikipedia.org/wiki/Smoothstep

APP_DIR = Path(__file__).resolve().parent      # the folder this file lives in
MUSIC_DIR = APP_DIR / "music"                  # where your songs are kept
CACHE_DIR = APP_DIR / ".moodwave_cache"        # saved analysis results (replays are instant)
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg"}  # the file types MoodWave will accept

FPS = 60                  # how many times per second the screen redraws
N_BANDS = 26              # how many pitch "slices" (low to high) we measure
ANALYSIS_SR = 24000       # songs are read at this quality: fast, and plenty for visuals
HOP = 400                 # 24000 / 400 = 60 measurements for every second of music

# The pastel palette (baby blue, lilac, baby pink, mint). The INK colours are
# for text; SHADOW is the cool tint used on the shaded side of the cloud.
WHITE = (255, 255, 255)
INK = (52, 64, 110)
INK_SOFT = (128, 141, 182)
BLUE = (143, 178, 245)
BLUE_DEEP = (106, 140, 236)
PINK = (250, 187, 214)
PINK_DEEP = (240, 146, 188)
LILAC = (200, 182, 242)
MINT = (168, 230, 208)
SKY = (172, 210, 250)
SHADOW = (150, 160, 226)
PALETTE_STOPS = [SKY, LILAC, PINK, MINT, SKY]  # loops back to the start


# Turns any number into a colour on the pastel palette, blending smoothly
# from blue to lilac to pink to mint and back round to blue.
def palette(u):
    """Smoothly blend around the pastel palette. u is any number (wraps)."""
    u = u % 1.0
    n = len(PALETTE_STOPS) - 1
    x = u * n
    i = min(int(x), n - 1)
    f = x - i
    f = f * f * (3 - 2 * f)
    a, b = PALETTE_STOPS[i], PALETTE_STOPS[i + 1]
    return tuple(a[k] + (b[k] - a[k]) * f for k in range(3))


# ==========================================================================
# AUDIO ANALYSIS
# ==========================================================================
#
#   Before a song plays, MoodWave "listens" to the whole file once and notes down,
#   60 times a second, how strong the bass, middle and high notes are, how loud it
#   is, where the beats land, and the tempo (BPM). While the song plays, the
#   visuals just look these numbers up; results are saved so replays are instant.
#
# Helpful resources:
#   - librosa on GitHub (the audio-analysis library doing the heavy lifting)
#     https://github.com/librosa/librosa
#   - librosa documentation
#     https://librosa.org/doc/
#   - librosa.stft - splitting sound into frequencies
#     https://librosa.org/doc/latest/generated/librosa.stft.html
#   - librosa.feature.melspectrogram - grouping frequencies into bands
#     https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
#   - librosa.onset.onset_strength - spotting beats and hits
#     https://librosa.org/doc/latest/generated/librosa.onset.onset_strength.html
#   - librosa.beat.beat_track - estimating the tempo (BPM)
#     https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html
#   - The mel scale (why pitch bands are spaced the way they are)
#     https://en.wikipedia.org/wiki/Mel_scale
#   - python-soundfile on GitHub (reads the audio files)
#     https://github.com/bastibe/python-soundfile
#   - Python threading (running the analysis in the background)
#     https://docs.python.org/3/library/threading.html
#   - Example project: librosa mel spectrogram turned into visuals
#     https://github.com/MathioLucas/ASCIIMusicVisualizer
#   - Example project: different frequencies drive different effects
#     https://github.com/wardzin/animusic

# Holds the recorded numbers for one song and can answer the question:
# "what was the music doing at second X?"
class Analysis:
    """Per-frame features for one song (spectrum, bass/mid/treble, beats)."""

    def __init__(self, spectrum, bass, mid, treble, rms, onset, tempo, duration):
        self.spectrum = spectrum
        self.bass, self.mid, self.treble = bass, mid, treble
        self.rms, self.onset = rms, onset
        self.tempo = float(tempo)
        self.duration = float(duration)
        self.n_frames = spectrum.shape[1]
        self.frame_rate = ANALYSIS_SR / HOP

    def features_at(self, t):
        i = int(max(0.0, t) * self.frame_rate)
        i = min(i, self.n_frames - 1)
        return {
            "spectrum": self.spectrum[:, i],
            "bass": float(self.bass[i]),
            "mid": float(self.mid[i]),
            "treble": float(self.treble[i]),
            "rms": float(self.rms[i]),
            "onset": float(self.onset[i]),
            "tempo": self.tempo,
        }


# Tiny helper: trims or pads a list of numbers to an exact length, so all the
# measurements line up frame for frame.
def _fit(x, n):
    x = np.asarray(x).ravel()
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)), mode="edge")


# Works out the file name for a song's saved analysis. The name changes if the
# song file changes, so an edited song gets analysed again.
def _cache_path(path):
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}|v2"
    return CACHE_DIR / (hashlib.md5(key.encode()).hexdigest() + ".npz")


# The heart of the analysis: reads a song and returns an Analysis. The numbered
# steps inside follow what happens, in order.
def analyze_file(path, progress=lambda s: None):
    """Analyse a song (or load the saved result if we've seen it before)."""
    # Step 0: have we already analysed this exact file? If so, reuse the result.
    cache = _cache_path(path)
    if cache.exists():
        try:
            d = np.load(cache)
            progress("Loaded from memory")
            return Analysis(d["spectrum"], d["bass"], d["mid"], d["treble"],
                            d["rms"], d["onset"], d["tempo"], d["duration"])
        except Exception:
            pass

    progress("Warming up the audio engine")
    import librosa  # imported here so the window opens instantly

    # Step 1: read the song as mono audio at a lower quality (faster, and more than
    # enough detail for visuals).
    progress("Reading your song")
    y, sr = librosa.load(str(path), sr=ANALYSIS_SR, mono=True)
    duration = len(y) / sr

    # Step 2: split the sound into frequencies, then measure how strong the bass
    # (20-250 Hz), middle (250-2000 Hz) and treble (2000-8000 Hz) are over time.
    progress("Listening to the frequencies")
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    nf = S.shape[1]

    def band(lo, hi):
        idx = (freqs >= lo) & (freqs < hi)
        return S[idx].mean(axis=0)

    def norm(x):
        lo, hi = np.percentile(x, 5), np.percentile(x, 97)
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

    bass = norm(band(20, 250))
    mid = norm(band(250, 2000))
    treble = norm(band(2000, 8000))

    # Step 3: squash the spectrum into 26 pitch bands (low to high). These drive the
    # different areas of the cloud.
    progress("Mapping the spectrum")
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=N_BANDS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    lo = np.percentile(mel_db, 5, axis=1, keepdims=True)
    hi = np.percentile(mel_db, 98, axis=1, keepdims=True)
    spectrum = np.clip((mel_db - lo) / (hi - lo + 1e-6), 0, 1)

    # Step 4: measure loudness, spot sudden hits (onsets) and estimate the tempo.
    progress("Finding the beat")
    rms = norm(_fit(librosa.feature.rms(S=S, frame_length=2048)[0], nf))
    onset_env = _fit(librosa.onset.onset_strength(S=mel_db, sr=sr, hop_length=HOP), nf)
    onset = np.clip(onset_env / (np.percentile(onset_env, 95) + 1e-6), 0, 1)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0]) or 120.0

    # Step 5: save everything so this song loads instantly next time.
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez(cache, spectrum=spectrum, bass=bass, mid=mid, treble=treble,
                 rms=rms, onset=onset, tempo=tempo, duration=duration)
    except Exception:
        pass

    progress("Almost there")
    return Analysis(spectrum, bass, mid, treble, rms, onset, tempo, duration)


# Runs analyze_file() on a background thread, so the window keeps animating
# (instead of freezing) while a song is being analysed.
class AnalysisJob(threading.Thread):
    """Runs the analysis in the background so the window stays alive."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path
        self.status = "Getting ready"
        self.result = None
        self.error = None
        self.cancelled = False

    def run(self):
        try:
            self.result = analyze_file(self.path, self._progress)
        except Exception as e:  # shown to the user on the loading screen
            self.error = str(e) or repr(e)

    def _progress(self, text):
        self.status = text


# ==========================================================================
# LIBRARY
# ==========================================================================
#
#   Finds the songs you've added: it checks the music folder (and the folder this
#   file is in), keeps only formats MoodWave can play, and sorts them A-Z. A few
#   tiny helpers tidy song names, open the music folder and turn seconds into "m:ss".
#
# Helpful resources:
#   - Python pathlib (listing and checking files)
#     https://docs.python.org/3/library/pathlib.html
#   - pygame events, including DROPFILE for drag-and-drop
#     https://www.pygame.org/docs/ref/event.html
#   - Example project: pygame + librosa player where users add their own songs
#     https://github.com/audreywl/baclaudio

# Collects every supported audio file from the music folder and the app folder,
# ignores hidden files, and sorts them A-Z.
def scan_library():
    found = {}
    for folder in (MUSIC_DIR, APP_DIR):
        try:
            for p in folder.iterdir():
                if (p.is_file() and p.suffix.lower() in AUDIO_EXTS
                        and not p.name.startswith(".")):
                    found.setdefault(p.name.lower(), p)
        except OSError:
            pass
    return sorted(found.values(), key=lambda p: p.stem.lower())


# "Hype_Boy.mp3" becomes "Hype Boy": the friendly name shown on screen.
def pretty_name(p):
    return p.stem.replace("_", " ").strip()


# Opens a folder in Finder (Mac), Explorer (Windows) or the file manager (Linux).
def open_folder(path):
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


# Turns seconds into m:ss, for example 83 becomes "1:23".
def format_time(sec):
    m, s = divmod(int(max(0, sec)), 60)
    return f"{m}:{s:02d}"


# ==========================================================================
# DRAWING HELPERS
# ==========================================================================
#
#   Small reusable tools for drawing the interface: fonts, text, the soft
#   near-white background, and a blurry oval used for the cloud's shadow. The rest
#   of the code leans on these so it stays short and readable.
#
# Helpful resources:
#   - pygame-ce on GitHub (the library that opens the window and draws)
#     https://github.com/pygame-community/pygame-ce
#   - pygame-ce documentation
#     https://pyga.me/docs
#   - NumPy broadcasting (how the background gradient is built quickly)
#     https://numpy.org/doc/1.20/user/basics.broadcasting.html

# Looks up fonts once and remembers them. Tries a few nice system fonts first
# (Avenir, Helvetica...) and falls back to whatever is available.
class Fonts:
    names = "avenirnext,avenir,helveticaneue,helvetica,arial,dejavusans"
    _cache = {}

    @classmethod
    def get(cls, size, bold=False):
        key = (max(8, int(size)), bold)
        f = cls._cache.get(key)
        if f is None:
            f = pygame.font.SysFont(cls.names, key[0], bold=bold)
            cls._cache[key] = f
        return f


# Draws text on the screen and returns where it landed (handy for lining things up).
def draw_text(surf, text, font, color, **anchor):
    img = font.render(text, True, color)
    rect = img.get_rect(**anchor)
    surf.blit(img, rect)
    return rect


# Shortens text that is too long to fit and adds "..." on the end.
def ellipsize(text, font, max_w):
    if font.size(text)[0] <= max_w:
        return text
    while len(text) > 1 and font.size(text + "...")[0] > max_w:
        text = text[:-1]
    return text.rstrip() + "..."


# Returns a size multiplier based on the window size, so the layout shrinks
# and grows with the window.
def ui_scale(size):
    return max(0.55, min(size) / 750.0)


_bg_cache = {}


# Builds the page background: white fading to a very pale blue, with a faint
# pink and blue glow in two corners. Built small, scaled up, then remembered.
def background(size):
    """Near-white page with the faintest pastel glow in two corners."""
    surf = _bg_cache.get(size)
    if surf is None:
        w, h = size
        sw, sh = max(8, w // 4), max(8, h // 4)  # build small, scale up = smooth
        t = np.linspace(0, 1, sh, dtype=np.float32)[:, None, None]
        top = np.array((255, 255, 255), np.float32)
        bot = np.array((244, 248, 255), np.float32)
        arr = np.broadcast_to(top * (1 - t) + bot * t, (sh, sw, 3)).copy()
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)

        def glow(cx, cy, r, color, strength):
            nonlocal arr
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
            a = (np.clip(1 - d, 0, 1) ** 2 * strength)[..., None]
            arr = arr * (1 - a) + np.array(color, np.float32) * a

        glow(sw * 0.10, sh * 0.98, max(sw, sh) * 0.60, PINK, 0.22)
        glow(sw * 0.92, sh * 0.04, max(sw, sh) * 0.55, SKY, 0.26)
        small = pygame.surfarray.make_surface(arr.astype(np.uint8).swapaxes(0, 1))
        surf = pygame.transform.smoothscale(small, (w, h)).convert()
        if len(_bg_cache) > 4:
            _bg_cache.clear()
        _bg_cache[size] = surf
    return surf


_ellipse_cache = {}


# Makes a blurry-edged oval (used for the cloud's soft shadow) and remembers it.
def soft_ellipse(w, h, color, max_alpha):
    key = (w, h, color, max_alpha)
    s = _ellipse_cache.get(key)
    if s is None:
        ys, xs = np.mgrid[0:h, 0:w]
        x = (xs - w / 2 + 0.5) / (w / 2)
        y = (ys - h / 2 + 0.5) / (h / 2)
        a = np.clip(1 - np.sqrt(x * x + y * y), 0, 1) ** 1.4 * max_alpha
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[..., :3] = color
        rgba[..., 3] = a.astype(np.uint8)
        s = pygame.image.frombuffer(rgba.tobytes(), (w, h), "RGBA").convert_alpha()
        if len(_ellipse_cache) > 20:
            _ellipse_cache.clear()
        _ellipse_cache[key] = s
    return s


# ==========================================================================
# THE CLOUD SPHERE
# ==========================================================================
#
#   The cloud is a sphere made of hundreds of soft, overlapping puffs. The music
#   pushes, resizes and tints them: low notes swell the bottom, high notes shimmer
#   at the top, and beats make the whole cloud breathe in. It swells quickly but
#   settles slowly, which is what makes it feel like a cloud.
#
# Helpful resources:
#   - Spreading points evenly over a sphere (the Fibonacci sphere)
#     https://extremelearning.com.au/evenly-distributing-points-on-a-sphere/
#   - Fibonacci lattice, a short and friendly explanation
#     https://www.johndcook.com/blog/2023/08/12/fibonacci-lattice/
#   - 3D rotation (how the sphere is spun)
#     https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
#   - 3D projection (flattening 3D points onto the screen)
#     https://en.wikipedia.org/wiki/3D_projection
#   - Perspective projection, step by step (Scratchapixel)
#     https://scratchapixel.com/lessons/3d-basic-rendering/computing-pixel-coordinates-of-3d-point/perspective-projection.html
#   - Exponential smoothing (quick to swell, slow to settle)
#     https://en.wikipedia.org/wiki/Exponential_smoothing
#   - NumPy on GitHub (all the maths is done with arrays)
#     https://github.com/numpy/numpy
#   - Example project: numpy + pygame audio visualisers
#     https://github.com/nate-h/Whirling
#   - Example project: particle visualiser reacting to bass, mids and highs
#     https://github.com/tmc3221/AudioVisualizer

# Tuning for the puffs:
#   PAL_STEPS - how many shades we pre-draw from the palette
#   SIZE_STEP - how coarsely puff sizes are rounded, so pictures can be reused
#   SHADE_MIX - how much the lighting tints a puff (shadow side to lit side)
#   LIGHT     - the direction the light comes from (upper left, towards you)
PAL_STEPS = 24
SIZE_STEP = 1.16
# How each puff is tinted by lighting: (tint colour, amount)
SHADE_MIX = [(SHADOW, 0.50), (SHADOW, 0.20), (WHITE, 0.0), (WHITE, 0.28)]
LIGHT = np.array([-0.45, 0.60, 0.66])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


# Draws ONE puff as a small soft-edged picture, with a light top-left highlight
# and a slightly darker underside. A handful are drawn once and reused, which
# keeps the animation fast.
def make_puff(color, shade, r):
    """One fluffy puff sprite: a soft-edged pastel blob with a bright top-left
    highlight and a slightly shaded underside (what makes it read as cloud)."""
    tint, amt = SHADE_MIX[shade]
    base = np.array(color, np.float32) * (1 - amt) + np.array(tint, np.float32) * amt
    n = 2 * r
    ys, xs = np.mgrid[0:n, 0:n]
    x = (xs - r + 0.5) / r
    y = (ys - r + 0.5) / r
    d = np.sqrt(x * x + y * y)
    a = np.clip((1 - d) / 0.72, 0, 1)
    alpha = a * a * (3 - 2 * a) * 0.90            # soft, fluffy rim
    hl = np.clip(1 - np.sqrt((x + 0.32) ** 2 + (y + 0.34) ** 2) / 1.0, 0, 1) ** 1.4
    low = np.clip((x + y) * 0.5 + 0.15, 0, 1) * np.clip(d, 0, 1)   # underside
    col = base[None, None, :]
    col = col * (1 - 0.20 * hl[..., None]) + 255.0 * 0.20 * hl[..., None]
    sh = np.array(SHADOW, np.float32)[None, None, :]
    col = col * (1 - 0.22 * low[..., None]) + sh * 0.22 * low[..., None]
    rgba = np.dstack([col, alpha * 255]).clip(0, 255).astype(np.uint8)
    return pygame.image.frombuffer(rgba.tobytes(), (n, n), "RGBA").convert_alpha()


# A cheap, smooth wobble pattern. Give it a position and the time and it returns
# a slowly changing bump or dip, which makes the surface roll gently like a
# real cumulus cloud.
def billow(p, t):
    """Smooth, slow, lumpy noise (-1..1) so the cloud billows like a cumulus."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    n = (np.sin(x * 2.3 + t * 0.9) * np.cos(y * 1.9 - t * 0.6)
         + np.sin(y * 3.1 - t * 1.1) * np.cos(z * 2.4 + t * 0.7)
         + np.sin(z * 2.0 + t * 0.5) * np.cos(x * 2.7 - t * 0.8))
    return n / 3.0


# Gentle pretend 'music' used on the title and loading screens, so the cloud
# still looks alive when nothing is playing.
def idle_features(t):
    """Gentle fake 'music' so the title/loading screens feel alive."""
    k = np.arange(N_BANDS)
    return {
        "spectrum": 0.40 + 0.20 * np.sin(t * 0.9 + k * 0.45),
        "bass": 0.35 + 0.18 * math.sin(t * 0.7),
        "mid": 0.4, "treble": 0.4 + 0.15 * math.sin(t * 0.5 + 1.0),
        "rms": 0.35 + 0.10 * math.sin(t * 0.55),
        "onset": 0.0, "tempo": 90.0,
    }


# The cloud itself. update() reacts to the music; draw() paints it.
class CloudSphere:
    """A sphere made of many soft, overlapping puffs of pastel colour.

    Low notes swell the bottom of the cloud, high notes make the top
    shimmer, beats make the whole cloud inhale, and everything eases
    back slowly (fast attack, slow release) so it feels soft and billowy.
    """

    # Sets up the puffs once: spreads ~420 of them evenly over a sphere (some inside,
    # for fullness), and gives each a size, a colour along a smooth gradient, and a
    # pitch "band" (puffs near the bottom listen to bass, near the top to treble).
    def __init__(self, n=420, seed=11):
        rng = np.random.default_rng(seed)
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        theta = np.pi * (1 + 5 ** 0.5) * i
        self.dirs = np.stack([np.cos(theta) * np.sin(phi), np.cos(phi),
                              np.sin(theta) * np.sin(phi)], axis=1)
        shell = rng.random(n) < 0.78
        self.rad = np.where(shell, rng.uniform(0.84, 1.0, n), rng.uniform(0.35, 0.80, n))
        self.size = rng.uniform(0.21, 0.31, n) * np.where(shell, 1.0, 1.25)
        self.phase = rng.uniform(0, 2 * np.pi, n)
        d = self.dirs
        # smooth gradient across the sphere, with just a touch of randomness
        self.cmix = (0.5 + 0.30 * d[:, 1] + 0.22 * d[:, 0]
                     + 0.12 * np.sin(d[:, 2] * 3.0) + rng.normal(0, 0.025, n)) % 1.0
        height = (self.dirs[:, 1] + 1) / 2  # 0 = bottom, 1 = top
        band = height * (N_BANDS - 1) + rng.normal(0, 3.0, n)
        self.band = np.clip(np.round(band), 0, N_BANDS - 1).astype(int)

        self.yaw = float(rng.uniform(0, 360))
        self.pitch = 0.0
        self.flow = 0.0
        self.clock_t = 0.0
        self.pal_phase = 0.0
        self.mood = 0.5
        self.pulse = 0.0
        self.s_spec = np.full(N_BANDS, 0.3)
        self.s_bass = self.s_mid = self.s_treble = self.s_rms = 0.3
        self.sprites = {}

    # ---- update ---------------------------------------------------------
    # Runs every frame. Reads the latest music numbers and eases the cloud toward
    # them (quick to swell, slow to settle), then nudges the rotation, the slow
    # "billowing" motion and the colour drift forward a tiny step.
    def update(self, dt, f):
        dt = min(max(dt, 0.0), 0.1)
        up = 1 - math.exp(-16 * dt)     # quick to swell...
        down = 1 - math.exp(-3.0 * dt)  # ...slow to settle, like a cloud

        def sm(cur, target):
            return cur + (target - cur) * np.where(np.asarray(target) > cur, up, down)

        self.s_spec = sm(self.s_spec, np.asarray(f["spectrum"], float))
        self.s_bass = float(sm(self.s_bass, f["bass"]))
        self.s_mid = float(sm(self.s_mid, f["mid"]))
        self.s_treble = float(sm(self.s_treble, f["treble"]))
        self.s_rms = float(sm(self.s_rms, f["rms"]))

        if f["onset"] > 0.55:
            self.pulse = max(self.pulse, min(1.0, f["onset"]))
        self.pulse *= math.exp(-4.5 * dt)

        tempo_n = f.get("tempo", 100.0) / 120.0
        speed = (7.0 + 5.0 * tempo_n) * (1 + 0.5 * self.s_treble + 0.6 * self.pulse)
        self.yaw = (self.yaw + speed * dt) % 360
        self.clock_t += dt
        self.pitch = 12 * math.sin(self.clock_t * 0.13)
        self.flow += dt * (0.30 + 0.8 * self.s_rms + 0.7 * self.pulse)

        target_mood = 0.5 + 0.5 * (self.s_treble - self.s_bass)
        self.mood += (target_mood - self.mood) * (1 - math.exp(-0.5 * dt))
        self.pal_phase += dt * 0.015

    # ---- draw -----------------------------------------------------------
    # Fetches the right pre-drawn puff (by colour, shading and size), drawing it
    # first if it hasn't been made yet.
    # Runs every frame. Works out where each puff is in 3D (pushed around by the
    # music), spins and lights the sphere, flattens it to 2D with perspective, then
    # paints the puffs from farthest to nearest so they overlap naturally.
    def _sprite(self, pal_i, shade, qi):
        key = (pal_i, shade, qi)
        s = self.sprites.get(key)
        if s is None:
            if len(self.sprites) > 1400:
                self.sprites.clear()
            r = max(3, int(SIZE_STEP ** qi))
            s = make_puff(palette(pal_i / PAL_STEPS), shade, r)
            self.sprites[key] = s
        return s

    def draw(self, surf, cx, cy, R):
        bass, rms, pulse = self.s_bass, self.s_rms, self.pulse
        spec = self.s_spec[self.band]

        # soft ground shadow
        sw = int(R * 2.3)
        sh = max(6, int(R * 0.34 * (1 - 0.15 * bass)))
        shadow = soft_ellipse(sw, sh, (176, 170, 226), 70)
        surf.blit(shadow, shadow.get_rect(center=(int(cx), int(cy + R * 1.38))))

        # billowing, audio-driven puff positions (object space)
        pos0 = self.dirs * self.rad[:, None]
        noise = billow(pos0, self.flow)
        r_eff = self.rad * (1 + 0.10 * bass + 0.18 * spec
                            + noise * (0.04 + 0.09 * rms) + 0.12 * pulse)
        pos = self.dirs * r_eff[:, None]
        ph = self.phase
        wob = 0.03 * np.sin(self.flow * 1.4 + ph)[:, None] * np.stack(
            [np.sin(ph * 1.7), np.cos(ph * 2.3), np.sin(ph * 3.1)], axis=1)
        pos = pos + wob

        # rotate + light + perspective
        ya, xa = math.radians(self.yaw), math.radians(self.pitch)
        cyw, syw, cxp, sxp = math.cos(ya), math.sin(ya), math.cos(xa), math.sin(xa)
        Ry = np.array([[cyw, 0, syw], [0, 1, 0], [-syw, 0, cyw]])
        Rx = np.array([[1, 0, 0], [0, cxp, -sxp], [0, sxp, cxp]])
        M = Rx @ Ry
        p = pos @ M.T
        L = (self.dirs @ M.T) @ LIGHT
        shade = np.digitize(L, [-0.15, 0.30, 0.70])

        cam = 3.6
        persp = cam / (cam - p[:, 2])
        sx = cx + p[:, 0] * R * persp
        sy = cy - p[:, 1] * R * persp
        rad_px = self.size * R * persp * (1 + 0.26 * spec + 0.14 * pulse + 0.07 * bass)
        qi = np.round(np.log(np.maximum(rad_px, 1.0)) / math.log(SIZE_STEP)).astype(int)

        u = (self.cmix * 0.80 + self.pal_phase + self.mood * 0.30) % 1.0
        pal = (u * PAL_STEPS).astype(int) % PAL_STEPS

        order = np.argsort(p[:, 2])  # far puffs first
        sx_l, sy_l = sx.tolist(), sy.tolist()
        pal_l, shade_l, qi_l = pal.tolist(), shade.tolist(), qi.tolist()
        rad_l = rad_px.tolist()
        for i in order.tolist():
            if rad_l[i] < 3:
                continue
            spr = self._sprite(pal_l[i], shade_l[i], qi_l[i])
            surf.blit(spr, (int(sx_l[i]) - spr.get_width() // 2,
                            int(sy_l[i]) - spr.get_height() // 2))


# ==========================================================================
# THE APP
# ==========================================================================
# 
#   The App class is the conductor. About 60 times a second it checks the mouse and
#   keyboard, draws whichever screen you're on (title, library, loading or playing),
#   and keeps the music and the cloud in sync. The method groups below are labelled
#   by job.
#
# Helpful resources:
#   - The game loop pattern (what run() and frame() are doing)
#     https://www.gameprogrammingpatterns.com/game-loop.html
#   - pygame.mixer.music (playing, pausing and timing the song)
#     https://pygame.org/docs/ref/music.html
#   - pygame events (mouse, keyboard, drag-and-drop)
#     https://www.pygame.org/docs/ref/event.html
#   - pygame-ce documentation
#     https://pyga.me/docs
#   - Python threading (the loading screen stays alive while analysing)
#     https://docs.python.org/3/library/threading.html
#   - Example project: pygame window that analyses a song, then plays it
#     https://github.com/audreywl/baclaudio
#   - Example project: music-reactive app with beat detection (librosa)
#     https://github.com/digidigital/openlightshow

# The whole application. Owns the window, the current screen, the library,
# the cloud and the music player.
class App:
    # Opens the window and sets the starting state: title screen, library loaded,
    # nothing playing.
    def __init__(self):
        MUSIC_DIR.mkdir(exist_ok=True)
        pygame.init()
        pygame.display.set_caption("MoodWave")
        try:
            pygame.mixer.init(frequency=44100)
        except pygame.error as e:
            print("Audio device problem:", e)

        self.windowed_size = (1100, 750)
        self.fullscreen = False
        self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.running = True
        self.t = 0.0

        self.cloud = CloudSphere()
        self.library = scan_library()
        self.selected = 0 if self.library else -1
        self.lib_cursor = max(self.selected, 0)
        self.lib_scroll = 0
        self.mouse_moved = False

        self.state = "title"          # title | library | loading | play
        self.return_state = "title"
        self.regions = []
        self._cursor = None
        self.toast = None

        self.job = None
        self.load_error = None
        self.analysis = None
        self.current_path = None
        self.started = False
        self.paused = False
        self.last_pos = 0.0
        self.play_clock = 0.0

    # ---------------------------------------------------------------- utils
    # Housekeeping: pop-up messages ("toasts"), refreshing the song list, adding
    # files you drag onto the window, and switching in and out of fullscreen.
    # Shows a short pop-up message for a few seconds.
    def say(self, text, seconds=3.0):
        self.toast = (text, time.time() + seconds)

    # Re-reads the song list and keeps the same song selected where possible.
    def refresh_library(self, select_path=None):
        keep = select_path or (self.library[self.selected] if 0 <= self.selected < len(self.library) else None)
        self.library = scan_library()
        self.selected = -1
        if keep is not None:
            for i, p in enumerate(self.library):
                if p.name == Path(keep).name:
                    self.selected = i
        if self.selected < 0 and self.library:
            self.selected = 0
        self.lib_cursor = max(0, min(self.lib_cursor, len(self.library) - 1))

    # Adds a dropped file (or every song in a dropped folder) by copying it into
    # the music folder.
    def add_song(self, path):
        p = Path(path)
        if p.is_dir():
            files = [f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS and f.is_file()]
            for f in files:
                self.add_song(f)
            if not files:
                self.say("No audio files found in that folder")
            return
        if p.suffix.lower() not in AUDIO_EXTS or not p.is_file():
            self.say("MoodWave can play mp3, wav, flac and ogg files")
            return
        dest = MUSIC_DIR / p.name
        try:
            if p.resolve().parent != MUSIC_DIR.resolve() and not dest.exists():
                shutil.copy2(p, dest)
                self.say(f"Added {pretty_name(p)}")
            else:
                self.say(f"{pretty_name(p)} is already in your library")
        except OSError as e:
            self.say(f"Couldn't add that file ({e})")
            return
        self.refresh_library(select_path=dest if dest.exists() else p)

    # Switches between the normal window and fullscreen.
    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)

    # ------------------------------------------------------------ navigation
    # Moving between screens. play_song() starts analysing in the background,
    # start_playback() begins the music once that's done, and stop_playback()
    # takes you back to where you came from.
    # Switches to the title screen.
    def go_title(self):
        self.state = "title"

    # Switches to the library screen.
    def go_library(self):
        self.state = "library"
        self.lib_cursor = max(self.selected, 0)
        self.lib_scroll = 0
        self._scroll_to_cursor()

    # The Play button: plays the selected song, or sends you to the library if it's empty.
    def press_play(self):
        if not self.library:
            self.say("Add a song to your library first")
            self.go_library()
        else:
            self.play_song(max(self.selected, 0))

    # Starts a song: kicks off the analysis in the background and shows the loading
    # screen.
    def play_song(self, index):
        if not (0 <= index < len(self.library)):
            return
        path = self.library[index]
        if not path.exists():
            self.say("That file has gone missing")
            self.refresh_library()
            return
        self.return_state = self.state if self.state in ("title", "library") else "title"
        self.selected = index
        self.current_path = path
        self.load_error = None
        self.analysis = None
        self.job = AnalysisJob(path)
        self.job.start()
        self.state = "loading"

    # Called once the analysis is done: starts the music and switches to the playing
    # screen.
    def start_playback(self):
        try:
            pygame.mixer.music.load(str(self.current_path))
            pygame.mixer.music.play()
        except pygame.error as e:
            self.load_error = f"This file couldn't be played ({e})"
            print(f"MoodWave: {self.load_error}")
            return
        self.state = "play"
        self.started = False
        self.paused = False
        self.last_pos = 0.0
        self.play_clock = 0.0

    # Stops the music and returns to the screen you came from (optionally with a message).
    def stop_playback(self, message=None):
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        if self.job:
            self.job.cancelled = True
            self.job = None
        self.state = self.return_state
        if message:
            self.say(message)

    # Pauses or resumes the music (and the cloud along with it).
    def toggle_pause(self):
        if self.state != "play":
            return
        self.paused = not self.paused
        if self.paused:
            pygame.mixer.music.pause()
        else:
            pygame.mixer.music.unpause()

    # ----------------------------------------------------------------- input
    # Everything you do with the mouse and keyboard: clicks, scrolling, drag-and-
    # drop and shortcuts. What each key does depends on the screen you're on.
    # Reads everything that happened since the last frame: closing the window, mouse
    # clicks and movement, scrolling, dropped files and key presses.
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.VIDEORESIZE and not self.fullscreen:
                self.windowed_size = (e.w, e.h)
            elif e.type == pygame.DROPFILE:
                self.add_song(e.file)
            elif e.type == pygame.MOUSEMOTION:
                self.mouse_moved = True
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for rect, cb in reversed(self.regions):
                    if rect.collidepoint(e.pos):
                        cb()
                        break
            elif e.type == pygame.MOUSEWHEEL and self.state == "library":
                self.lib_scroll -= e.y * 50
            elif e.type == pygame.KEYDOWN:
                self.on_key(e.key)

    # Keyboard shortcuts. What a key does depends on which screen you're on.
    def on_key(self, key):
        if key == pygame.K_f:
            self.toggle_fullscreen()
            return
        if self.state == "title":
            if key == pygame.K_ESCAPE:
                self.running = False
            elif key in (pygame.K_RETURN, pygame.K_SPACE):
                self.press_play()
            elif key == pygame.K_l:
                self.go_library()
        elif self.state == "library":
            if key == pygame.K_ESCAPE:
                self.go_title()
            elif key == pygame.K_DOWN and self.library:
                self.lib_cursor = min(self.lib_cursor + 1, len(self.library) - 1)
                self._scroll_to_cursor()
            elif key == pygame.K_UP and self.library:
                self.lib_cursor = max(self.lib_cursor - 1, 0)
                self._scroll_to_cursor()
            elif key == pygame.K_RETURN and self.library:
                self.play_song(self.lib_cursor)
        elif self.state == "loading":
            if key == pygame.K_ESCAPE:
                self.stop_playback()
        elif self.state == "play":
            if key == pygame.K_ESCAPE:
                self.stop_playback()
            elif key == pygame.K_SPACE:
                self.toggle_pause()

    # Works out where the song list goes on screen (size, position, row height) for
    # the current window size.
    def _lib_layout(self):
        w, h = self.screen.get_size()
        u = ui_scale((w, h))
        list_w = min(int(w * 0.8), int(820 * u))
        top, bottom = int(124 * u), h - int(124 * u)
        return {
            "u": u, "x": (w - list_w) // 2, "w": list_w, "top": top, "bottom": bottom,
            "row_h": int(62 * u), "gap": int(12 * u),
        }

    # Scrolls the library just enough to keep the highlighted song in view.
    def _scroll_to_cursor(self):
        L = self._lib_layout()
        y0 = self.lib_cursor * (L["row_h"] + L["gap"])
        view_h = L["bottom"] - L["top"]
        if y0 < self.lib_scroll:
            self.lib_scroll = y0
        elif y0 + L["row_h"] > self.lib_scroll + view_h:
            self.lib_scroll = y0 + L["row_h"] - view_h

    # --------------------------------------------------------------- widgets
    # Reusable interface pieces: the pill-shaped buttons and the small pop-up
    # message. Each button registers a clickable area, so clicks can be matched
    # to it on the next frame.
    # Draws a pill-shaped button (highlighted when the mouse is over it) and
    # registers it as clickable.
    def button(self, rect, label, kind, callback, icon=None):
        s = self.screen
        hover = rect.collidepoint(pygame.mouse.get_pos())
        r = rect.inflate(10, 6) if hover else rect
        pygame.draw.rect(s, (205, 216, 244), r.move(0, 4), border_radius=r.h // 2)
        if kind == "primary":
            fill, fg, border = ((122, 154, 240) if hover else BLUE_DEEP), WHITE, None
        else:
            fill, fg, border = ((244, 248, 255) if hover else WHITE), INK, BLUE
        pygame.draw.rect(s, fill, r, border_radius=r.h // 2)
        if border:
            pygame.draw.rect(s, border, r, width=3, border_radius=r.h // 2)

        img = Fonts.get(r.h * 0.40, True).render(label, True, fg)
        icon_w = int(r.h * 0.30) + 12 if icon else 0
        x0 = r.centerx - (icon_w + img.get_width()) / 2
        cy = r.centery
        if icon == "play":
            t = r.h * 0.17
            pygame.draw.polygon(s, fg, [(x0, cy - t), (x0, cy + t), (x0 + t * 1.7, cy)])
        elif icon == "back":
            t = r.h * 0.15
            pygame.draw.lines(s, fg, False,
                              [(x0 + t * 1.2, cy - t * 1.3), (x0, cy), (x0 + t * 1.2, cy + t * 1.3)], 3)
        s.blit(img, (x0 + icon_w, cy - img.get_height() / 2))
        self.regions.append((rect, callback))

    # Draws the pop-up message bubble, if there is one.
    def draw_toast(self, y):
        if not self.toast or time.time() > self.toast[1]:
            return
        w, h = self.screen.get_size()
        u = ui_scale((w, h))
        img = Fonts.get(18 * u, True).render(self.toast[0], True, INK)
        r = img.get_rect(center=(w // 2, y)).inflate(int(40 * u), int(20 * u))
        pygame.draw.rect(self.screen, WHITE, r, border_radius=r.h // 2)
        pygame.draw.rect(self.screen, PINK, r, 2, border_radius=r.h // 2)
        self.screen.blit(img, img.get_rect(center=r.center))

    # --------------------------------------------------------------- screens
    # One drawing method per screen: title, library, loading and playing (plus the
    # duration bar at the bottom of the playing screen).
    # Title screen: the idle cloud, the MoodWave logo, and the Play and Library buttons.
    def draw_title(self, dt):
        s = self.screen
        w, h = s.get_size()
        u = ui_scale((w, h))
        s.blit(background((w, h)), (0, 0))

        self.cloud.update(dt, idle_features(self.t))
        self.cloud.draw(s, w / 2, h * 0.27, min(w, h) * 0.135)

        tf = Fonts.get(104 * u, True)
        a = tf.render("Mood", True, BLUE_DEEP)
        b = tf.render("Wave", True, PINK_DEEP)
        sa = tf.render("Mood", True, (214, 222, 248))
        sb = tf.render("Wave", True, (250, 222, 236))
        x = (w - a.get_width() - b.get_width()) // 2
        y = int(h * 0.49)
        s.blit(sa, (x + 4, y + 5))
        s.blit(sb, (x + a.get_width() + 4, y + 5))
        s.blit(a, (x, y))
        s.blit(b, (x + a.get_width(), y))
        draw_text(s, "see your music as a cloud", Fonts.get(24 * u), INK_SOFT,
                  midtop=(w // 2, y + int(a.get_height() * 0.98)))

        bw, bh, gap = int(230 * u), int(64 * u), int(24 * u)
        by = int(h * 0.755)
        self.button(pygame.Rect(w // 2 - bw - gap // 2, by, bw, bh), "Play", "primary",
                    self.press_play, icon="play")
        self.button(pygame.Rect(w // 2 + gap // 2, by, bw, bh), "Library", "soft",
                    self.go_library)

        font = Fonts.get(18 * u)
        if self.library and self.selected >= 0:
            name = ellipsize(pretty_name(self.library[self.selected]), font, int(w * 0.7))
            draw_text(s, "Ready:  " + name, font, INK_SOFT, midtop=(w // 2, by + bh + int(30 * u)))
        else:
            draw_text(s, "Open the Library to add your first song", font, INK_SOFT,
                      midtop=(w // 2, by + bh + int(30 * u)))
        self.draw_toast(h - int(40 * u))

    # Library screen: a scrollable list of songs with hover highlight, a scroll bar
    # and the drag-and-drop hint.
    def draw_library(self, dt):
        s = self.screen
        w, h = s.get_size()
        L = self._lib_layout()
        u = L["u"]
        s.blit(background((w, h)), (0, 0))

        self.button(pygame.Rect(int(32 * u), int(28 * u), int(130 * u), int(46 * u)),
                    "Back", "soft", self.go_title, icon="back")
        draw_text(s, "Library", Fonts.get(46 * u, True), INK, midtop=(w // 2, int(24 * u)))
        n = len(self.library)
        draw_text(s, f"{n} song" + ("" if n == 1 else "s"), Fonts.get(18 * u), INK_SOFT,
                  midright=(w - int(36 * u), int(52 * u)))

        view = pygame.Rect(L["x"], L["top"], L["w"], L["bottom"] - L["top"])
        row_h, gap = L["row_h"], L["gap"]

        if n == 0:
            draw_text(s, "Your library is empty", Fonts.get(32 * u, True), INK,
                      center=(w // 2, view.centery - int(20 * u)))
            draw_text(s, "Drag and drop songs onto this window to add them",
                      Fonts.get(20 * u), INK_SOFT, center=(w // 2, view.centery + int(20 * u)))
        else:
            total = n * (row_h + gap)
            max_scroll = max(0, total - view.h)
            self.lib_scroll = min(max(self.lib_scroll, 0), max_scroll)
            mouse = pygame.mouse.get_pos()
            name_font = Fonts.get(23 * u, True)
            small = Fonts.get(16 * u)
            s.set_clip(view)
            for i, p in enumerate(self.library):
                y = view.top + i * (row_h + gap) - self.lib_scroll
                rect = pygame.Rect(view.x + 6, y + 4, view.w - 12, row_h)
                if rect.bottom < view.top or rect.top > view.bottom:
                    continue
                hit = rect.clip(view)
                if self.mouse_moved and hit.collidepoint(mouse):
                    self.lib_cursor = i
                active = i == self.lib_cursor
                chosen = i == self.selected
                rr = rect.inflate(10, 4) if active else rect
                pygame.draw.rect(s, (212, 222, 246), rr.move(0, 3), border_radius=rr.h // 2)
                pygame.draw.rect(s, (240, 246, 255) if active else WHITE, rr, border_radius=rr.h // 2)
                border = BLUE if active else (PINK if chosen else (222, 230, 248))
                pygame.draw.rect(s, border, rr, 2, border_radius=rr.h // 2)

                br = int(row_h * 0.29)
                bc = (rr.x + int(row_h * 0.52), rr.centery)
                pygame.draw.circle(s, palette(i * 0.17), bc, br)
                draw_text(s, str(i + 1), Fonts.get(br * 0.95, True), INK, center=bc)

                right_space = int(110 * u)
                name = ellipsize(pretty_name(p), name_font, rr.w - (bc[0] - rr.x) - br - 40 - right_space)
                draw_text(s, name, name_font, INK, midleft=(bc[0] + br + 18, rr.centery))
                if active:
                    tx = rr.right - int(row_h * 0.62) - int(34 * u)
                    t = row_h * 0.12
                    pygame.draw.polygon(s, BLUE_DEEP, [(tx, rr.centery - t), (tx, rr.centery + t),
                                                       (tx + t * 1.7, rr.centery)])
                    draw_text(s, "Play", Fonts.get(18 * u, True), BLUE_DEEP,
                              midleft=(tx + t * 1.7 + 10, rr.centery))
                elif chosen:
                    draw_text(s, "selected", small, INK_SOFT,
                              midright=(rr.right - int(row_h * 0.5), rr.centery))
                self.regions.append((hit, lambda i=i: self.play_song(i)))
            s.set_clip(None)

            if max_scroll > 0:
                th = max(30, int(view.h * view.h / total))
                ty = view.top + int((view.h - th) * self.lib_scroll / max_scroll)
                pygame.draw.rect(s, (214, 224, 246), (view.right + 6, ty, 5, th), border_radius=3)
        self.mouse_moved = False

        draw_text(s, "Drag and drop mp3, wav, flac or ogg files onto this window to add songs",
                  Fonts.get(17 * u), INK_SOFT, center=(w // 2, h - int(92 * u)))
        self.button(pygame.Rect(w // 2 - int(110 * u), h - int(72 * u), int(220 * u), int(44 * u)),
                    "Open music folder", "soft", lambda: open_folder(MUSIC_DIR))
        self.draw_toast(int(98 * u))

    # Loading screen: shown while a song is being analysed (or if it fails to load).
    def draw_loading(self, dt):
        s = self.screen
        w, h = s.get_size()
        u = ui_scale((w, h))
        s.blit(background((w, h)), (0, 0))

        job = self.job
        if job is not None:
            if job.error and not self.load_error:
                self.load_error = job.error
                print(f"MoodWave: couldn't analyse {self.current_path}: {job.error}")
            elif job.result is not None and not job.cancelled:
                self.analysis = job.result
                self.job = None
                self.start_playback()
                if self.state == "play":
                    return

        self.cloud.update(dt, idle_features(self.t * 1.6))
        self.cloud.draw(s, w / 2, h * 0.36, min(w, h) * 0.15)

        name = pretty_name(self.current_path) if self.current_path else ""
        if self.load_error:
            draw_text(s, "Hmm, that song didn't load", Fonts.get(36 * u, True), INK,
                      center=(w // 2, int(h * 0.67)))
            draw_text(s, "The file may be damaged, or isn't a format MoodWave can read.",
                      Fonts.get(18 * u), INK_SOFT, center=(w // 2, int(h * 0.67) + int(40 * u)))
            draw_text(s, "Press Esc to go back", Fonts.get(18 * u), INK_SOFT,
                      center=(w // 2, int(h * 0.67) + int(76 * u)))
            return

        dots = "." * (int(self.t * 2.5) % 4)
        draw_text(s, "Getting your cloud ready" + dots, Fonts.get(34 * u, True), INK,
                  center=(w // 2, int(h * 0.67)))
        draw_text(s, ellipsize(name, Fonts.get(22 * u), int(w * 0.8)), Fonts.get(22 * u), BLUE_DEEP,
                  center=(w // 2, int(h * 0.67) + int(42 * u)))
        status = job.status if job else ""
        draw_text(s, status, Fonts.get(17 * u), INK_SOFT, center=(w // 2, int(h * 0.67) + int(76 * u)))

        bw, bh = int(300 * u), max(6, int(8 * u))
        bx, byy = (w - bw) // 2, int(h * 0.67) + int(106 * u)
        pygame.draw.rect(s, (222, 230, 248), (bx, byy, bw, bh), border_radius=bh // 2)
        seg = int(bw * 0.3)
        pos = ((self.t * 0.9) % 1.4 - 0.2) * bw
        x0, x1 = max(bx, bx + int(pos)), min(bx + bw, bx + int(pos) + seg)
        if x1 > x0:
            pygame.draw.rect(s, PINK_DEEP, (x0, byy, x1 - x0, bh), border_radius=bh // 2)
        draw_text(s, "Esc to cancel", Fonts.get(15 * u), INK_SOFT, center=(w // 2, h - int(36 * u)))

    # How many seconds into the song we are, as reported by the music player.
    def song_time(self):
        pos = pygame.mixer.music.get_pos()
        if pos >= 0:
            self.last_pos = pos / 1000.0
        return self.last_pos

    # Playing screen: feeds the music numbers for this moment into the cloud, draws
    # it, and adds the song title, Back button and duration bar.
    def draw_play(self, dt):
        s = self.screen
        w, h = s.get_size()
        u = ui_scale((w, h))
        s.blit(background((w, h)), (0, 0))

        if pygame.mixer.music.get_busy():
            self.started = True
        if self.started and not self.paused and not pygame.mixer.music.get_busy():
            self.stop_playback(f"Finished {pretty_name(self.current_path)}")
            return

        a = self.analysis
        t = self.song_time()
        if not self.paused:
            self.cloud.update(dt, a.features_at(t))
            self.play_clock += dt
        cx, cy, R = w / 2, h * 0.45, min(w, h) * 0.175
        self.cloud.draw(s, cx, cy, R)
        self.regions.append((pygame.Rect(cx - R * 1.4, cy - R * 1.4, R * 2.8, R * 2.8), self.toggle_pause))

        self.button(pygame.Rect(int(28 * u), int(24 * u), int(120 * u), int(42 * u)),
                    "Back", "soft", self.stop_playback, icon="back")
        draw_text(s, "NOW PLAYING", Fonts.get(13 * u, True), INK_SOFT, midtop=(w // 2, int(22 * u)))
        draw_text(s, ellipsize(pretty_name(self.current_path), Fonts.get(24 * u, True), int(w * 0.6)),
                  Fonts.get(24 * u, True), INK, midtop=(w // 2, int(42 * u)))
        if self.paused:
            draw_text(s, "Paused", Fonts.get(22 * u, True), INK_SOFT,
                      center=(w // 2, int(cy + R * 1.75)))

        self.draw_progress(t, a.duration)
        if self.play_clock < 6 and not self.paused:
            alpha_text = "space: pause     esc: back     f: fullscreen"
            draw_text(s, alpha_text, Fonts.get(15 * u), INK_SOFT, center=(w // 2, h - int(18 * u)))
        self.draw_toast(int(104 * u))

    # The duration bar: elapsed time, a bar that fills (shifting from blue to pink)
    # as the song goes on, and the total length.
    def draw_progress(self, t, total):
        s = self.screen
        w, h = s.get_size()
        u = ui_scale((w, h))
        frac = min(1.0, max(0.0, t / total)) if total > 0 else 0.0
        bw = min(int(w * 0.6), 720)
        bh = max(6, int(7 * u))
        x, y = (w - bw) // 2, h - int(58 * u)
        pygame.draw.rect(s, (222, 230, 248), (x, y, bw, bh), border_radius=bh // 2)
        fw = int(bw * frac)
        color = tuple(int(BLUE[k] + (PINK_DEEP[k] - BLUE[k]) * frac) for k in range(3))
        if fw > 0:
            pygame.draw.rect(s, color, (x, y, max(fw, bh), bh), border_radius=bh // 2)
        kr = int(bh * 1.35)
        kc = (x + fw, y + bh // 2)
        pygame.draw.circle(s, WHITE, kc, kr)
        pygame.draw.circle(s, color, kc, kr, 3)
        f = Fonts.get(15 * u)
        draw_text(s, format_time(t), f, INK_SOFT, midright=(x - int(14 * u), y + bh // 2))
        draw_text(s, format_time(total), f, INK_SOFT, midleft=(x + bw + int(14 * u), y + bh // 2))

    # ------------------------------------------------------------- main loop
    # frame() does one frame of work: handle input, draw the current screen, update
    # the mouse cursor and show the result. run() repeats it until you quit.
    def frame(self, dt):
        self.screen = pygame.display.get_surface()
        self.handle_events()
        self.regions = []
        if self.state == "title":
            self.draw_title(dt)
        elif self.state == "library":
            self.draw_library(dt)
        elif self.state == "loading":
            self.draw_loading(dt)
            if self.state == "play":
                self.draw_play(0.0)
        elif self.state == "play":
            self.draw_play(dt)

        pos = pygame.mouse.get_pos()
        want = (pygame.SYSTEM_CURSOR_HAND if any(r.collidepoint(pos) for r, _ in self.regions)
                else pygame.SYSTEM_CURSOR_ARROW)
        if want != self._cursor:
            try:
                pygame.mouse.set_cursor(want)
                self._cursor = want
            except pygame.error:
                pass
        pygame.display.flip()

    def run(self):
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            self.t += dt
            self.frame(dt)
        pygame.quit()


# ==========================================================================
# START
# ==========================================================================
# 
#   Running `python3 moodwave.py` starts here: it creates the App and runs it
#   until you close the window.
#
# Helpful resources:
#   - Setup and how to run it on Mac and Windows: see README.md
#   - Python virtual environments (keeps this project's libraries tidy)
#     https://docs.python.org/3/library/venv.html

def main():
    App().run()


if __name__ == "__main__":
    main()
