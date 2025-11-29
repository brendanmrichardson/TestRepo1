"""Generate a headphone-friendly resonance breathing soundscape.

The script builds a layered stereo signal made from slow breathing motions,
a dark ambient bed, rain-like microtexture, and gentle minute markers.
Only numpy and the Python standard library are used.
"""

import argparse
import math
import random
import wave
from typing import List, Tuple

import numpy as np

# Audio constants
SAMPLE_RATE = 48_000
DEFAULT_DURATION = 600  # seconds
OUTPUT_FILENAME = "resonance_headphones.wav"


def moving_average(signal: np.ndarray, kernel_size: int) -> np.ndarray:
    """Efficient moving average using cumulative sum."""
    if kernel_size <= 1:
        return signal
    pad = kernel_size // 2
    padded = np.pad(signal, (pad, pad), mode="reflect")
    cumsum = np.cumsum(padded, dtype=float)
    cumsum[kernel_size:] = cumsum[kernel_size:] - cumsum[:-kernel_size]
    smoothed = cumsum[kernel_size - 1:] / kernel_size
    return smoothed[: signal.size]


def generate_breathing_phase(num_samples: int, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate breathing phase with mean-reverting frequency variation."""
    ctrl_rate = 50  # Hz
    step = max(1, int(sr / ctrl_rate))
    f_target = 0.1
    alpha = 0.05
    sigma = 0.002
    f_min, f_max = 0.08, 0.12

    f = f_target
    phase = 0.0
    phases = np.empty(num_samples)

    for i in range(0, num_samples, step):
        # Update frequency via mean-reverting stochastic process
        f += alpha * (f_target - f) + sigma * random.gauss(0.0, 1.0)
        f = max(f_min, min(f_max, f))
        # Integrate within the control block
        for j in range(step):
            idx = i + j
            if idx >= num_samples:
                break
            phase += 2 * math.pi * f / sr
            if phase >= 2 * math.pi:
                phase -= 2 * math.pi
            phases[idx] = phase
    phase01 = phases / (2 * math.pi)
    return phases, phase01


def mean_reverting_intervals(total_duration: float, target: float = 60.0, alpha: float = 0.1,
                             sigma: float = 0.8, min_v: float = 50.0, max_v: float = 70.0) -> List[float]:
    """Generate a series of times using a mean-reverting interval process."""
    times: List[float] = []
    t = 0.0
    interval = target
    while t < total_duration:
        times.append(t)
        interval += alpha * (target - interval) + sigma * random.gauss(0.0, 1.0)
        interval = max(min_v, min(max_v, interval))
        t += interval
    return times


def build_minute_envelope(num_samples: int, sr: int, duration: float) -> Tuple[np.ndarray, List[float]]:
    """Create overlapping Hann envelopes for slowly intensifying minute markers."""
    marker_times = mean_reverting_intervals(duration)
    env = np.zeros(num_samples)
    env_dur = 3.0
    env_samples = int(env_dur * sr)
    hann_win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(env_samples) / (env_samples - 1))

    base_intensity = 0.25
    step_intensity = 0.05
    max_intensity = 1.2

    for idx, t in enumerate(marker_times):
        center = int(t * sr)
        start = max(0, center - env_samples // 2)
        end = min(num_samples, start + env_samples)
        win = hann_win[: end - start]
        intensity = math.tanh(base_intensity + idx * step_intensity) * max_intensity
        env[start:end] += win * intensity

    return env, marker_times


def slow_lfo(length: int, sr: int, freq: float, phase: float = 0.0) -> np.ndarray:
    """Sine LFO."""
    t = np.arange(length) / sr
    return np.sin(2 * np.pi * freq * t + phase)


def generate_base_bed(num_samples: int, sr: int) -> np.ndarray:
    """Dark ambient bed using filtered noise and ultra-slow drift."""
    noise = np.random.normal(0, 1, num_samples)
    low = moving_average(noise, kernel_size=int(sr * 0.02))  # ~20 ms smoothing
    high = noise - moving_average(noise, kernel_size=int(sr * 0.002))
    bed = 0.65 * low + 0.25 * high
    drift = slow_lfo(num_samples, sr, freq=0.003)
    bed *= 0.9 + 0.1 * drift
    bed = bed / (np.max(np.abs(bed)) + 1e-6)
    return bed


def generate_breathing_layer(num_samples: int, sr: int, phase: np.ndarray) -> np.ndarray:
    """Airy pad keyed to the breathing phase for amplitude and brightness."""
    noise = np.random.normal(0, 1, num_samples)
    low = moving_average(noise, kernel_size=int(sr * 0.015))
    high = noise - moving_average(noise, kernel_size=int(sr * 0.0015))

    brightness = 0.6 + 0.3 * np.sin(phase + 0.2)
    amp = 0.7 + 0.3 * np.sin(phase - math.pi / 2)
    brightness = np.clip(brightness, 0.3, 0.95)
    amp = np.clip(amp, 0.4, 1.0)

    layer = (1 - brightness) * low + brightness * high
    layer *= amp
    layer = layer / (np.max(np.abs(layer)) + 1e-6)
    return layer


def generate_rain(num_samples: int, sr: int, breathing_phase01: np.ndarray) -> np.ndarray:
    """Sparse microtexture with breathing- and LFO-driven density shifts."""
    base_rate = 14.0  # events per second
    lfo_freq = random.uniform(0.01, 0.03)
    rate_depth = 0.35
    breath_influence = 0.1 * np.sin(2 * np.pi * breathing_phase01)

    step = int(sr / 1000)  # 1 ms steps
    impulses = []
    for start in range(0, num_samples, step):
        t = start / sr
        lfo = math.sin(2 * math.pi * lfo_freq * t)
        rate = base_rate * (1 + rate_depth * lfo + breath_influence[start])
        rate = max(2.0, rate)
        lam = rate * (step / sr)
        count = np.random.poisson(lam)
        if count <= 0:
            continue
        for _ in range(count):
            offset = random.randint(0, step - 1)
            idx = start + offset
            if idx < num_samples:
                amp = 0.15 / (1 + rate)
                impulses.append((idx, amp))

    env_ms = 25
    env_samples = int(sr * env_ms / 1000)
    if env_samples < 2:
        env_samples = 2
    env = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(env_samples) / (env_samples - 1))

    signal = np.zeros(num_samples)
    for idx, amp in impulses:
        end = min(num_samples, idx + env_samples)
        seg = env[: end - idx]
        signal[idx:end] += amp * seg

    low = moving_average(signal, kernel_size=int(sr * 0.0015))
    high = signal - low
    rain = high * (1.0 + 0.2 * np.sin(2 * np.pi * breathing_phase01 + 0.3))
    if np.max(np.abs(rain)) > 0:
        rain = rain / (np.max(np.abs(rain)) + 1e-6)
    rain *= 0.25
    return rain


def apply_minute_effects(signal: np.ndarray, minute_env: np.ndarray, marker_times: List[float], sr: int) -> np.ndarray:
    """Subtle brightening, swells, and harmonic dust at minute markers."""
    max_env = np.max(minute_env) + 1e-6
    bright = signal - moving_average(signal, kernel_size=int(sr * 0.002))
    enhanced = signal + 0.3 * minute_env * bright / max_env

    # Whispery harmonic cluster riding on the marker envelope
    tone_freqs = [220.0, 330.0, 440.0]
    t = np.arange(signal.size) / sr
    tone = np.zeros_like(signal)
    for i, f in enumerate(tone_freqs):
        tone += 0.1 * np.sin(2 * np.pi * f * t + i * 0.5)
    tone *= (minute_env / max_env) * 0.1
    enhanced += tone

    for idx, mt in enumerate(marker_times):
        # Per-marker energy nibble so each swell feels slightly unique
        local_gain = 1.0 + 0.05 * math.sin(idx * 1.1)
        start = int(mt * sr)
        end = min(signal.size, start + int(3.0 * sr))
        enhanced[start:end] *= local_gain

    return enhanced


def stereo_field(mono: np.ndarray, breathing_phase: np.ndarray, sr: int) -> np.ndarray:
    """Breathing-linked width modulation with decorrelated side content."""
    width = 0.5 + 0.4 * np.sin(breathing_phase + 0.8)
    side_noise = np.random.normal(0, 1, mono.size)
    side_noise = moving_average(side_noise, kernel_size=int(sr * 0.002))
    side_noise = side_noise / (np.max(np.abs(side_noise)) + 1e-6)

    side = side_noise * width * 0.4
    left = mono + side
    right = mono - side

    shelf = mono - moving_average(mono, kernel_size=int(sr * 0.001))
    left += 0.1 * shelf
    right += 0.1 * shelf

    stereo = np.stack([left, right], axis=1)
    max_val = np.max(np.abs(stereo)) + 1e-6
    stereo /= max_val
    return stereo


def normalize_audio(stereo: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(stereo))
    if peak == 0:
        return stereo
    return stereo * (0.99 / peak)


def write_wav(filename: str, data: np.ndarray, sr: int) -> None:
    """Write stereo float data to 16-bit PCM WAV."""
    int_data = np.clip(data, -1.0, 1.0)
    int_data = (int_data * 32767).astype('<i2')
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_data.tobytes())


def render(duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Render the full stereo soundscape."""
    num_samples = int(duration * sr)
    phase, phase01 = generate_breathing_phase(num_samples, sr)

    base_bed = generate_base_bed(num_samples, sr)
    breathing_layer = generate_breathing_layer(num_samples, sr, phase)
    rain = generate_rain(num_samples, sr, phase01)

    minute_env, marker_times = build_minute_envelope(num_samples, sr, duration)

    mono = 0.55 * base_bed + 0.45 * breathing_layer + 0.25 * rain
    mono = mono / (np.max(np.abs(mono)) + 1e-6)

    mono = apply_minute_effects(mono, minute_env, marker_times, sr)

    stereo = stereo_field(mono, phase, sr)
    stereo = normalize_audio(stereo)
    return stereo


def main():
    parser = argparse.ArgumentParser(description="Generate headphone-optimised resonance breathing soundscape.")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help="Duration in seconds (default: 600)")
    parser.add_argument("--output", type=str, default=OUTPUT_FILENAME,
                        help="Output WAV filename")
    args = parser.parse_args()

    stereo = render(args.duration, SAMPLE_RATE)
    write_wav(args.output, stereo, SAMPLE_RATE)
    print(f"Wrote {args.output} ({args.duration:.1f}s, {SAMPLE_RATE} Hz, stereo 16-bit PCM)")


if __name__ == "__main__":
    main()
