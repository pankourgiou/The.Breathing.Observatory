import pygame
import sounddevice as sd
import numpy as np
import soundfile as sf
import random, math, sys, time

# ---------- CONFIG ----------
SR = 44100
BLOCK = 1024
MASTER_VOL = 0.05
BPM = 70
BREATH_PERIOD = 60.0 / BPM * 2.0
BASE_NOTE = 220.0
NOTE_INTERVAL = 2.5
MAX_NOTES = 6
RECORD_FILE = "output_recording.wav"

SCALES = {
    "enigmatic": [0, 1, 4, 6, 8, 10, 11],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor_pent": [0, 3, 5, 7, 10],
}

# ---------- AUDIO ----------
class Note:
    def __init__(self, freq, dur, sr):
        self.freq = freq
        self.dur = dur
        self.sr = sr
        self.t = 0
        self.env = self._make_env()
        self.finished = False

    def _make_env(self):
        L = int(self.dur * self.sr)
        attack = np.linspace(0, 1, L // 8)
        sustain = np.ones(L // 2)
        release = np.linspace(1, 0, L - len(attack) - len(sustain))
        return np.concatenate([attack, sustain, release])

    def generate(self, frames):
        if self.finished:
            return np.zeros(frames)
        start = self.t
        end = self.t + frames
        if end >= len(self.env):
            end = len(self.env)
            self.finished = True
        n = end - start
        t = (np.arange(n) + start) / self.sr
        tri = 2 * np.abs(2 * (self.freq * t - np.floor(self.freq * t + 0.5))) - 1
        sine = np.sin(2 * np.pi * self.freq * t)
        out = (0.6 * tri + 0.4 * sine) * self.env[start:end]
        res = np.zeros(frames)
        res[:n] = out
        self.t += n
        return res


class AmbientSynth:
    def __init__(self, sr=SR):
        self.sr = sr
        self.t = 0
        self.notes = []
        self.delay_buf = np.zeros(int(2 * sr))
        self.dp = 0
        self.record_buffer = []
        self.last_note_time = 0
        self.scale_name = "lydian"
        self.scale = SCALES[self.scale_name]

    def set_scale(self, name):
        if name in SCALES:
            self.scale_name = name
            self.scale = SCALES[name]
            print(f"🎶 Switched to {name} scale")

    def trigger(self):
        freq = BASE_NOTE * 2 ** (random.choice(self.scale) / 12)
        dur = random.uniform(3, 8)
        self.notes.append(Note(freq, dur, self.sr))

    def generate(self, frames):
        t = (np.arange(frames) + self.t) / self.sr
        self.t += frames
        breath = 0.6 + 0.4 * (0.5 * (1 + np.sin(2 * np.pi * t / BREATH_PERIOD)))

        if (time.time() - self.last_note_time) > NOTE_INTERVAL:
            if len(self.notes) < MAX_NOTES:
                self.trigger()
                self.last_note_time = time.time()

        mix = np.zeros(frames)
        alive = []
        for n in self.notes:
            mix += n.generate(frames)
            if not n.finished:
                alive.append(n)
        self.notes = alive

        # delay
        buf = self.delay_buf
        dp = self.dp
        dlen = len(buf)
        delay_samps = int(0.4 * self.sr)
        wet = np.zeros_like(mix)
        for i in range(frames):
            rp = (dp - delay_samps) % dlen
            val = buf[rp]
            wet[i] = val
            buf[dp] = mix[i] + 0.4 * val
            dp = (dp + 1) % dlen
        self.dp = dp

        out = (mix + 0.5 * wet + 0.002 * np.sin(2 * np.pi * 110 * t)) * MASTER_VOL
        out = np.clip(out, -1, 1)
        stereo = np.stack([out, out], axis=-1).astype(np.float32)
        self.record_buffer.append(stereo.copy())
        return stereo


synth = AmbientSynth()

def audio_cb(outdata, frames, time, status):
    if status:
        print(status)
    out = synth.generate(frames)
    outdata[:] = out


# ---------- VISUALS ----------
class Shape:
    def __init__(self, w, h):
        self.cx = w / 2
        self.cy = h / 2
        self.radius = random.randint(80, min(w, h) // 3)
        self.angle = random.uniform(0, math.pi * 2)
        self.speed = random.uniform(-0.3, 0.3)
        self.size = random.uniform(50, 200)
        self.color = [random.randint(100, 255) for _ in range(3)]
        self.shape_type = random.choice(["circle", "triangle", "square"])

    def draw(self, surf, phase):
        angle = self.angle + phase * self.speed
        x = self.cx + self.radius * math.cos(angle)
        y = self.cy + self.radius * math.sin(angle)
        s = int(self.size * (0.8 + 0.2 * math.sin(phase)))
        c = [min(255, int(v * (0.6 + 0.4 * abs(math.sin(phase))))) for v in self.color]
        if self.shape_type == "circle":
            pygame.draw.circle(surf, c, (int(x), int(y)), s // 2, 2)
        elif self.shape_type == "square":
            rect = pygame.Rect(x - s // 2, y - s // 2, s, s)
            pygame.draw.rect(surf, c, rect, 2)
        elif self.shape_type == "triangle":
            pts = [
                (x + math.cos(angle) * s, y + math.sin(angle) * s),
                (x + math.cos(angle + 2.1) * s, y + math.sin(angle + 2.1) * s),
                (x + math.cos(angle - 2.1) * s, y + math.sin(angle - 2.1) * s),
            ]
            pygame.draw.polygon(surf, c, pts, 2)


# ---------- MAIN ----------
pygame.init()
info = pygame.display.Info()
WIN = (info.current_w, info.current_h)
screen = pygame.display.set_mode(WIN, pygame.FULLSCREEN)
clock = pygame.time.Clock()
shapes = [Shape(*WIN) for _ in range(12)]

stream = sd.OutputStream(callback=audio_cb, samplerate=SR, channels=2, blocksize=BLOCK, dtype="float32")
stream.start()

running = True
time_acc = 0
print("🎧 Running... press 1–4 to change scales, ESC to quit and save recording.")

while running:
    dt = clock.tick(60) / 1000
    time_acc += dt
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_1: synth.set_scale("enigmatic")
            elif e.key == pygame.K_2: synth.set_scale("lydian")
            elif e.key == pygame.K_3: synth.set_scale("major")
            elif e.key == pygame.K_4: synth.set_scale("minor_pent")

    phase = time_acc / 3
    breath_phase = 0.5 * (1 + math.sin(2 * math.pi * time_acc / BREATH_PERIOD))

    bg = (int(10 + 40 * breath_phase), int(5 + 20 * breath_phase), int(40 + 80 * breath_phase))
    screen.fill(bg)
    for s in shapes:
        s.draw(screen, phase)

    pygame.display.flip()

# ---------- SHUTDOWN ----------
stream.stop()
stream.close()
pygame.quit()

if synth.record_buffer:
    audio_data = np.concatenate(synth.record_buffer)
    sf.write(RECORD_FILE, audio_data, SR)
    print(f"✅ Saved recording '{RECORD_FILE}'")
else:
    print("⚠️ No audio recorded.")
sys.exit()
