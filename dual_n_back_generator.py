"""
Dual n-back session generator producing non-interactive MP4 videos with audio using Piper TTS digit variants.
"""
import os
import math
import json
import random
import datetime
import subprocess
import shutil
from typing import List, Dict, Tuple

import numpy as np
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    VideoClip,
    ColorClip,
    concatenate_videoclips,
)
from moviepy.audio.AudioClip import AudioArrayClip

# -----------------------------------------------------------------------------
# Configuration constants (safe to tweak)
# -----------------------------------------------------------------------------
# Core task configuration
N_BACK = 3
BASE_TRIAL_DURATION = 2.0
SESSION_DURATION_SECONDS = 300
NUM_BATCH_FILES = 3

# Match probability scaling
SCALE_FACTOR = 1.10

# Video configuration (iPhone 13 Pro landscape)
VIDEO_WIDTH = 2532
VIDEO_HEIGHT = 1170
FPS = 30

# Piper TTS configuration
PIPER_EXE = "piper"
PIPER_MODEL_PATH = "PATH/TO/VOICE_MODEL.onnx"
PIPER_BASE_LENGTH_SCALE = 1.1
PIPER_BASE_NOISE_SCALE = 0.6
PIPER_BASE_NOISE_W_SCALE = 0.4
PIPER_VARIANTS_PER_DIGIT = 100

# Timing "aliveness"
TIMING_JITTER_FRACTION = 0.07
MIN_TRIAL_DURATION_FACTOR = 0.7

# Audio "aliveness"
BASE_DIGIT_VOLUME = 0.7
VOLUME_JITTER_FRACTION = 0.1
MIN_VOLUME = 0.5
MAX_VOLUME = 0.9

# Visual size & spatiality
MIN_RADIUS_FRACTION = 0.30
MAX_RADIUS_FRACTION = 0.60
SIZE_JITTER_FRACTION = 0.05
SCREEN_MARGIN_FRACTION = 0.05

# "Aliveness" state dynamics
NUM_SEGMENTS = 3
SEGMENT_BOUNDARIES = None
PACE_MEAN_REVERSION = 0.1
SALIENCE_MEAN_REVERSION = 0.1
SPATIALITY_MEAN_REVERSION = 0.1
STATE_NOISE_STD = 0.05

# Randomness
RANDOM_SEED = None

# Paths and filenames
OUTPUT_DIR = "output"
AUDIO_DIGIT_DIR = "audio_digits"
SUMMARY_SUFFIX = "_summary.txt"

# Other constants
TIME_OVERSHOOT_TOLERANCE = 5.0
BEEP_FREQ = 1000
BEEP_DURATION = 0.2
BEEP_VOLUME = 0.4

DIGIT_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}
COLOURS = ["red", "green", "blue", "yellow"]

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def ensure_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIGIT_DIR, exist_ok=True)


def all_digit_variants_exist() -> bool:
    for d in range(1, 10):
        for v in range(1, PIPER_VARIANTS_PER_DIGIT + 1):
            path = os.path.join(AUDIO_DIGIT_DIR, f"digit_{d}_var_{v}.wav")
            if not os.path.isfile(path):
                return False
    return True


def generate_digit_variants_with_piper():
    """Generate all digit audio variants using Piper TTS."""
    if not PIPER_MODEL_PATH or "PATH/TO" in PIPER_MODEL_PATH:
        raise RuntimeError("PIPER_MODEL_PATH is not set. Update the configuration before running.")
    if not shutil.which(PIPER_EXE):
        raise RuntimeError(f"Piper executable '{PIPER_EXE}' not found on PATH.")

    metadata = {}
    for d in range(1, 10):
        text = DIGIT_WORDS[d]
        for v in range(1, PIPER_VARIANTS_PER_DIGIT + 1):
            eps_len = random.uniform(-0.1, 0.1)
            eps_noise = random.uniform(-0.2, 0.2)
            eps_w = random.uniform(-0.2, 0.2)
            length_scale = clamp(PIPER_BASE_LENGTH_SCALE * (1.0 + eps_len), 0.8, 1.4)
            noise_scale = clamp(PIPER_BASE_NOISE_SCALE * (1.0 + eps_noise), 0.2, 1.0)
            noise_w_scale = clamp(PIPER_BASE_NOISE_W_SCALE * (1.0 + eps_w), 0.2, 1.0)
            seed = random.randint(0, 1_000_000)

            output_path = os.path.join(AUDIO_DIGIT_DIR, f"digit_{d}_var_{v}.wav")
            cmd = [
                PIPER_EXE,
                "-m",
                PIPER_MODEL_PATH,
                "--length-scale",
                str(length_scale),
                "--noise-scale",
                str(noise_scale),
                "--noise-w-scale",
                str(noise_w_scale),
                "--seed",
                str(seed),
                "-f",
                output_path,
            ]
            print(f"Generating digit {d} variant {v}/{PIPER_VARIANTS_PER_DIGIT} ...")
            try:
                subprocess.run(cmd, input=text.encode("utf-8"), check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"Piper generation failed for digit {d} variant {v}: {exc}") from exc
            metadata[os.path.basename(output_path)] = {
                "digit": d,
                "variant": v,
                "length_scale": length_scale,
                "noise_scale": noise_scale,
                "noise_w_scale": noise_w_scale,
                "seed": seed,
            }
    with open(os.path.join(AUDIO_DIGIT_DIR, "digit_variants_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# -----------------------------------------------------------------------------
# Session simulation
# -----------------------------------------------------------------------------

def derive_segment_boundaries() -> List[Tuple[float, float]]:
    if SEGMENT_BOUNDARIES:
        return SEGMENT_BOUNDARIES
    seg_len = SESSION_DURATION_SECONDS / NUM_SEGMENTS
    boundaries = []
    start = 0.0
    for _ in range(NUM_SEGMENTS):
        end = start + seg_len
        boundaries.append((start, end))
        start = end
    return boundaries


def segment_setpoints(boundaries: List[Tuple[float, float]]):
    setpoints = []
    for idx, (s, e) in enumerate(boundaries):
        if idx == 0:
            setpoints.append((0.95, 0.9, 0.4))
        elif idx == 1:
            setpoints.append((1.05, 1.1, 0.7))
        else:
            setpoints.append((1.0, 1.0, 0.5))
    return setpoints


def interpolate_setpoint(current_time: float, boundaries, setpoints):
    for idx, (start, end) in enumerate(boundaries):
        if start <= current_time <= end:
            # Smooth transition near boundaries
            trans = 10.0
            if idx < len(setpoints) - 1:
                next_start, _ = boundaries[idx + 1]
                if end - current_time < trans:
                    frac = (trans - (end - current_time)) / trans
                    base = setpoints[idx]
                    nxt = setpoints[idx + 1]
                    return tuple(base[i] * (1 - frac) + nxt[i] * frac for i in range(3))
            if idx > 0 and current_time - start < trans:
                prev = setpoints[idx - 1]
                frac = (trans - (current_time - start)) / trans
                base = setpoints[idx]
                return tuple(base[i] * (1 - frac) + prev[i] * frac for i in range(3))
            return setpoints[idx]
    return setpoints[-1]


def update_state(state: float, setpoint: float, mean_reversion: float) -> float:
    return state + mean_reversion * (setpoint - state) + random.gauss(0.0, STATE_NOISE_STD)


def sample_duration(pace_state: float) -> float:
    base_jitter = random.uniform(-TIMING_JITTER_FRACTION, TIMING_JITTER_FRACTION)
    jitter_factor = 1.0 + base_jitter + 0.1 * (pace_state - 1.0)
    jitter_factor = max(jitter_factor, MIN_TRIAL_DURATION_FACTOR)
    return BASE_TRIAL_DURATION * jitter_factor


def choose_circle_params(salience_state: float, spatiality_state: float):
    base_diameter_min = MIN_RADIUS_FRACTION * VIDEO_WIDTH
    base_diameter_max = MAX_RADIUS_FRACTION * VIDEO_WIDTH
    base_radius_min = base_diameter_min / 2.0
    base_radius_max = base_diameter_max / 2.0

    size_jitter = SIZE_JITTER_FRACTION * (1 + 0.2 * (salience_state - 1.0))
    r_min = base_radius_min * (1 + random.uniform(-size_jitter, size_jitter))
    r_max = base_radius_max * (1 + random.uniform(-size_jitter, size_jitter))
    if r_max < r_min:
        r_max, r_min = r_min, r_max

    margin_x = SCREEN_MARGIN_FRACTION * VIDEO_WIDTH
    margin_y = SCREEN_MARGIN_FRACTION * VIDEO_HEIGHT
    cx_center = VIDEO_WIDTH / 2.0
    cy_center = VIDEO_HEIGHT / 2.0

    # spatiality: 0 => central, 1 => peripheral
    max_offset_x = cx_center - margin_x - r_max
    max_offset_y = cy_center - margin_y - r_max
    distance_scale = clamp(spatiality_state, 0.0, 1.5)
    angle = random.uniform(0, 2 * math.pi)
    radius_factor = distance_scale * random.uniform(0.4, 1.0)
    dx = max_offset_x * radius_factor * math.cos(angle)
    dy = max_offset_y * radius_factor * math.sin(angle)
    cx = clamp(cx_center + dx, margin_x + r_max, VIDEO_WIDTH - margin_x - r_max)
    cy = clamp(cy_center + dy, margin_y + r_max, VIDEO_HEIGHT - margin_y - r_max)

    distance_from_center = math.hypot(cx - cx_center, cy - cy_center)
    max_distance = math.hypot(cx_center - margin_x, cy_center - margin_y)
    distance_normalized = distance_from_center / max_distance if max_distance > 0 else 0.0

    return r_min, r_max, cx, cy, distance_normalized, r_max / base_radius_max


def compute_volume(salience_state: float) -> float:
    volume = BASE_DIGIT_VOLUME * salience_state * (1.0 + random.uniform(-VOLUME_JITTER_FRACTION, VOLUME_JITTER_FRACTION))
    return clamp(volume, MIN_VOLUME, MAX_VOLUME)


def simulate_session(random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed % (2**32 - 1))

    boundaries = derive_segment_boundaries()
    setpoints = segment_setpoints(boundaries)

    pace_state = 1.0
    salience_state = 1.0
    spatiality_state = 1.0

    history: List[Dict] = []
    trials: List[Dict] = []
    trials_since_last_match = 0
    total_time = 0.0

    while total_time < SESSION_DURATION_SECONDS + TIME_OVERSHOOT_TOLERANCE:
        pace_sp, sal_sp, spat_sp = interpolate_setpoint(total_time, boundaries, setpoints)
        pace_state = update_state(pace_state, pace_sp, PACE_MEAN_REVERSION)
        salience_state = update_state(salience_state, sal_sp, SALIENCE_MEAN_REVERSION)
        spatiality_state = update_state(spatiality_state, spat_sp, SPATIALITY_MEAN_REVERSION)

        duration = sample_duration(pace_state)

        is_eligible = len(history) >= N_BACK
        audio_match = visual_match = False
        if not is_eligible:
            digit = random.randint(1, 9)
            digit_variant = random.randint(1, PIPER_VARIANTS_PER_DIGIT)
            colour = random.choice(COLOURS)
            is_match = False
        else:
            ref_idx = len(history) - N_BACK
            ref_digit = history[ref_idx]["digit"]
            ref_digit_variant = history[ref_idx]["digit_variant"]
            ref_colour = history[ref_idx]["colour"]
            p_start_audio = 1.0 / 9.0
            p_start_visual = 1.0 / 4.0
            p_match_audio = clamp(p_start_audio * (SCALE_FACTOR ** trials_since_last_match), 0.0, 0.99)
            p_match_visual = clamp(p_start_visual * (SCALE_FACTOR ** trials_since_last_match), 0.0, 0.99)
            audio_match = random.random() < p_match_audio
            visual_match = random.random() < p_match_visual
            if audio_match:
                digit = ref_digit
                digit_variant = ref_digit_variant
            else:
                choices = [x for x in range(1, 10) if x != ref_digit]
                digit = random.choice(choices)
                digit_variant = random.randint(1, PIPER_VARIANTS_PER_DIGIT)
            if visual_match:
                colour = ref_colour
            else:
                colour_choices = [c for c in COLOURS if c != ref_colour]
                colour = random.choice(colour_choices)
            is_match = audio_match and visual_match

        trial = {
            "digit": digit,
            "digit_variant": digit_variant,
            "colour": colour,
            "duration": duration,
            "is_match": is_match,
            "pace_state": pace_state,
            "salience_state": salience_state,
            "spatiality_state": spatiality_state,
            "start_time": total_time,
        }

        time_added = duration + (1.0 if is_match else 0.0)
        if total_time + time_added > SESSION_DURATION_SECONDS + TIME_OVERSHOOT_TOLERANCE:
            break

        r_min, r_max, cx, cy, dist_norm, radius_factor = choose_circle_params(salience_state, spatiality_state)
        volume = compute_volume(salience_state)
        trial.update({
            "r_min": r_min,
            "r_max": r_max,
            "cx": cx,
            "cy": cy,
            "distance_norm": dist_norm,
            "radius_factor": radius_factor,
            "volume": volume,
        })
        trials.append(trial)
        total_time += time_added

        if is_match:
            history = []
            trials_since_last_match = 0
        else:
            history.append(trial)
            if is_eligible:
                trials_since_last_match += 1
    session_info = {
        "trials": trials,
        "total_time": total_time,
        "boundaries": boundaries,
        "setpoints": setpoints,
    }
    return session_info


# -----------------------------------------------------------------------------
# Audio composition
# -----------------------------------------------------------------------------

def build_beep_clip(start_time: float):
    sample_rate = 44100
    t = np.linspace(0, BEEP_DURATION, int(sample_rate * BEEP_DURATION), endpoint=False)
    waveform = (np.sin(2 * np.pi * BEEP_FREQ * t) * BEEP_VOLUME).astype(np.float32)
    arr = waveform.reshape((-1, 1))
    return AudioArrayClip(arr, fps=sample_rate).set_start(start_time)


def build_audio_clip_from_trials(trials: List[Dict]):
    clips = []
    for trial in trials:
        audio_path = os.path.join(AUDIO_DIGIT_DIR, f"digit_{trial['digit']}_var_{trial['digit_variant']}.wav")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Missing audio file: {audio_path}")
        try:
            digit_clip = AudioFileClip(audio_path).volumex(trial["volume"]).set_start(trial["start_time"])
            clips.append(digit_clip)
        except Exception as exc:
            raise RuntimeError(f"Failed to load audio clip {audio_path}") from exc
        if trial["is_match"]:
            beep_start = trial["start_time"] + 0.75 * trial["duration"]
            clips.append(build_beep_clip(beep_start))
    if not clips:
        return None
    composite = CompositeAudioClip(clips)
    return composite


# -----------------------------------------------------------------------------
# Video composition helpers
# -----------------------------------------------------------------------------

def colour_to_rgb(colour: str) -> Tuple[float, float, float]:
    mapping = {
        "red": (220, 60, 60),
        "green": (60, 170, 80),
        "blue": (60, 90, 210),
        "yellow": (240, 200, 80),
    }
    return mapping.get(colour, (128, 128, 128))


def make_circle_frame_generator(trial: Dict):
    duration = trial["duration"]
    r_min = trial["r_min"]
    r_max = trial["r_max"]
    cx = trial["cx"]
    cy = trial["cy"]
    colour = trial["colour"]
    is_match = trial["is_match"]
    salience_state = trial["salience_state"]
    base_color = np.array(colour_to_rgb(colour), dtype=np.float32) / 255.0

    yy, xx = np.mgrid[0:VIDEO_HEIGHT, 0:VIDEO_WIDTH]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    ring_phase = random.uniform(0, 2 * math.pi)
    ring_freq = random.uniform(6, 10)
    ring_strength = 0.05 + 0.05 * (salience_state - 1.0)

    def make_frame(t):
        frame = np.ones((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.float32)
        radius = r_min + (r_max - r_min) * min(max(t / max(duration - 0.2, 0.001), 0.0), 1.0)
        alpha = 1.0
        if t >= duration - 0.2 and t < duration - 0.1:
            alpha = 1.0 - (t - (duration - 0.2)) / 0.1
        elif t >= duration - 0.1:
            alpha = 0.0
        mask = dist <= radius
        if np.any(mask) and alpha > 0:
            norm_dist = np.zeros_like(dist)
            norm_dist[mask] = dist[mask] / radius
            # radial gradient
            gradient = 0.9 + 0.1 * np.cos(norm_dist * math.pi)
            # rings
            rings = 1 + ring_strength * np.sin(norm_dist * ring_freq * math.pi + ring_phase)
            color_field = base_color * gradient[..., None] * rings[..., None]
            color_field = np.clip(color_field, 0, 1)
            frame[mask] = (1 - alpha) * frame[mask] + alpha * color_field[mask]

        if is_match and 0.75 * duration <= t < 0.75 * duration + 0.2:
            border_thickness = int(0.02 * VIDEO_HEIGHT)
            inset_x = int(0.1 * VIDEO_WIDTH)
            inset_y = int(0.1 * VIDEO_HEIGHT)
            frame[inset_y:inset_y + border_thickness, inset_x:VIDEO_WIDTH - inset_x, :] = base_color
            frame[VIDEO_HEIGHT - inset_y - border_thickness:VIDEO_HEIGHT - inset_y, inset_x:VIDEO_WIDTH - inset_x, :] = base_color
            frame[inset_y:VIDEO_HEIGHT - inset_y, inset_x:inset_x + border_thickness, :] = base_color
            frame[inset_y:VIDEO_HEIGHT - inset_y, VIDEO_WIDTH - inset_x - border_thickness:VIDEO_WIDTH - inset_x, :] = base_color
        return (frame * 255).astype(np.uint8)

    return make_frame


def build_video_clip_from_trials(trials: List[Dict]):
    clips = []
    for trial in trials:
        frame_gen = make_circle_frame_generator(trial)
        clip = VideoClip(frame_gen, duration=trial["duration"])
        clips.append(clip)
        if trial["is_match"]:
            clips.append(ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=(255, 255, 255), duration=1.0))
    if not clips:
        return None
    return concatenate_videoclips(clips)


# -----------------------------------------------------------------------------
# Summary file writer
# -----------------------------------------------------------------------------

def label_variation(value: float, target: float, tolerance: float) -> str:
    if value > target * (1 + tolerance):
        return "[HIGH]"
    if value < target * (1 - tolerance):
        return "[LOW]"
    return "[OK]"


def write_summary_file(path: str, video_filename: str, session_info: Dict, random_seed):
    trials = session_info["trials"]
    durations = [t["duration"] for t in trials]
    volumes = [t["volume"] for t in trials]
    radius_factors = [t["radius_factor"] for t in trials]
    distances = [t["distance_norm"] for t in trials]
    num_matches = sum(1 for t in trials if t["is_match"])

    trials_between = []
    since_last = 0
    for t in trials:
        if len(trials_between) == 0 and since_last == 0 and not t["is_match"]:
            since_last += 1
        elif t["is_match"]:
            trials_between.append(since_last)
            since_last = 0
        else:
            since_last += 1
    if since_last > 0:
        trials_between.append(since_last)

    mean_trials_between = sum(trials_between) / len(trials_between) if trials_between else 0
    max_trials_without = max(trials_between) if trials_between else 0

    mean_duration = sum(durations) / len(durations) if durations else 0
    duration_sd = float(np.std(durations)) if durations else 0
    min_duration = min(durations) if durations else 0
    max_duration = max(durations) if durations else 0

    mean_volume = sum(volumes) / len(volumes) if volumes else 0
    volume_sd = float(np.std(volumes)) if volumes else 0

    mean_radius_factor = sum(radius_factors) / len(radius_factors) if radius_factors else 0
    radius_factor_sd = float(np.std(radius_factors)) if radius_factors else 0

    mean_distance = sum(distances) / len(distances) if distances else 0
    distance_sd = float(np.std(distances)) if distances else 0

    content = []
    content.append("Dual n-back session summary")
    content.append("---------------------------")
    content.append(f"File: {video_filename}")
    content.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content.append(f"Random seed: {random_seed}")
    content.append("")
    content.append("Core parameters")
    content.append("---------------")
    content.append(f"N_BACK: {N_BACK}")
    content.append(f"BASE_TRIAL_DURATION: {BASE_TRIAL_DURATION:.2f} s")
    content.append(f"TARGET_SESSION_DURATION: {SESSION_DURATION_SECONDS} s")
    content.append(f"ACTUAL_SESSION_DURATION: {session_info['total_time']:.1f} s")
    content.append(f"NUM_TRIALS: {len(trials)}")
    content.append(f"SCALE_FACTOR: {SCALE_FACTOR}")
    content.append(f"AUDIO_VARIANTS_PER_DIGIT: {PIPER_VARIANTS_PER_DIGIT}")
    content.append("AUDIO_MATCH_VARIANT_POLICY: reuse_n_back_variant")
    content.append("")
    content.append("Match statistics")
    content.append("----------------")
    content.append(f"NUM_MATCHES: {num_matches}")
    content.append(f"MEAN_TRIALS_BETWEEN_MATCHES: {mean_trials_between:.1f}")
    content.append(f"MAX_TRIALS_WITHOUT_MATCH: {max_trials_without}")
    content.append("NOTE: Values are indicative only. [OK]")
    content.append("")
    content.append("Timing variability")
    content.append("------------------")
    content.append(f"MEAN_TRIAL_DURATION: {mean_duration:.2f} s  {label_variation(mean_duration, BASE_TRIAL_DURATION, 0.07)}")
    content.append(f"TRIAL_DURATION_SD: {duration_sd:.2f} s    {label_variation(duration_sd, BASE_TRIAL_DURATION * 0.1, 0.5)}")
    content.append(f"MIN_TRIAL_DURATION: {min_duration:.2f} s")
    content.append(f"MAX_TRIAL_DURATION: {max_duration:.2f} s")
    content.append("")
    content.append("Volume variability (digits)")
    content.append("---------------------------")
    content.append(f"MEAN_VOLUME (0–1): {mean_volume:.2f}      {label_variation(mean_volume, BASE_DIGIT_VOLUME, 0.2)}")
    content.append(f"VOLUME_SD: {volume_sd:.2f}             {label_variation(volume_sd, 0.05, 0.6)}")
    content.append("")
    content.append("Spatial and size variability")
    content.append("----------------------------")
    content.append(f"MEAN_RADIUS_FACTOR (vs base): {mean_radius_factor:.2f}   {label_variation(mean_radius_factor, 1.0, 0.15)}")
    content.append(f"RADIUS_FACTOR_SD: {radius_factor_sd:.2f}              {label_variation(radius_factor_sd, 0.05, 0.6)}")
    content.append(f"MEAN_DISTANCE_FROM_CENTRE (0–1): {mean_distance:.2f}  {label_variation(mean_distance, 0.35, 0.4)}")
    content.append(f"DISTANCE_SD: {distance_sd:.2f}")
    content.append("")
    content.append("Segment-level setpoints")
    content.append("-----------------------")
    for idx, ((start, end), sp) in enumerate(zip(session_info["boundaries"], session_info["setpoints"])):
        content.append(f"Segment {idx + 1}: t={start:.1f}s to t={end:.1f}s")
        content.append(f"  pace_setpoint: {sp[0]:.2f}")
        content.append(f"  salience_setpoint: {sp[1]:.2f}")
        content.append(f"  spatiality_setpoint: {sp[2]:.2f}")
        content.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main():
    ensure_directories()
    if RANDOM_SEED is not None:
        random_seed = RANDOM_SEED
    else:
        random_seed = random.randint(0, 1_000_000_000)
    random.seed(random_seed)
    np.random.seed(random_seed % (2**32 - 1))

    if not all_digit_variants_exist():
        try:
            generate_digit_variants_with_piper()
        except Exception as exc:
            print(f"Error generating digit variants: {exc}")
            return

    for batch in range(1, NUM_BATCH_FILES + 1):
        print(f"Simulating session {batch}/{NUM_BATCH_FILES} ...")
        session_info = simulate_session(random_seed + batch)
        trials = session_info["trials"]
        video_filename = f"{N_BACK}-back_{BASE_TRIAL_DURATION:g}s_F{batch}.mp4"
        summary_filename = f"{N_BACK}-back_{BASE_TRIAL_DURATION:g}s_F{batch}{SUMMARY_SUFFIX}"
        video_path = os.path.join(OUTPUT_DIR, video_filename)
        summary_path = os.path.join(OUTPUT_DIR, summary_filename)

        print("Building audio track ...")
        audio_clip = build_audio_clip_from_trials(trials)
        print("Building video track ... (this may take a while)")
        video_clip = build_video_clip_from_trials(trials)
        if video_clip is None:
            print("No trials generated; skipping file creation.")
            continue
        if audio_clip is not None:
            video_clip = video_clip.set_audio(audio_clip)
        total_duration = session_info["total_time"]
        try:
            video_clip.write_videofile(
                video_path,
                fps=FPS,
                codec="libx264",
                audio_codec="aac",
                bitrate="3000k",
                verbose=False,
                logger=None,
            )
        except Exception as exc:
            print(f"Failed to write video file: {exc}")
            continue
        print(f"Wrote {video_path} ({total_duration:.1f}s)")
        write_summary_file(summary_path, video_filename, session_info, random_seed + batch)
        print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
