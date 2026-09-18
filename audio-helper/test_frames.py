#!/usr/bin/env python3
"""
Parse one or more ghelper-audio binary frames from stdin and emit one JSON
object per line for the test harness to consume with `jq`.

Each frame is exactly 2596 bytes:
    16 B  header   (magic, version, seq, flags)
    20 B  scalars  (vad_prob, rms_in_db, rms_out_db, reduction_db, tracked_pitch_hz)
  2048 B  waveforms (256 in + 256 out, float32 each)
   512 B  spectra  ( 64 in +  64 out, float32 each)

Usage:
    ghelper-audio > stream.bin
    test_frames.py < stream.bin           # emit JSON lines
    test_frames.py --limit 3 < stream.bin # only first 3 frames
    test_frames.py --skip-waveform < ...  # omit big arrays
"""
import argparse
import json
import struct
import sys

WAVEFORM_SAMPLES = 256
SPECTRUM_BINS = 64
HDR = 4 * 4 + 5 * 4  # 4x uint32 + 5x float32
WAV_BYTES = WAVEFORM_SAMPLES * 4
SPEC_BYTES = SPECTRUM_BINS * 4
SPEC0 = HDR + 2 * WAV_BYTES
PKT = SPEC0 + 2 * SPEC_BYTES  # 2596
MAGIC = 0x47484146  # "GHAF"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="emit at most N frames (0 = unlimited)")
    ap.add_argument("--skip-waveform", action="store_true",
                    help="omit waveform_in/out arrays from JSON")
    ap.add_argument("--skip-spectrum", action="store_true",
                    help="omit spectrum_in/out arrays from JSON")
    args = ap.parse_args()

    raw = sys.stdin.buffer.read()
    n = len(raw) // PKT
    emitted = 0
    for i in range(n):
        p = raw[i * PKT:(i + 1) * PKT]
        magic, ver, seq, flags, vad, rin, rout, red, pitch = struct.unpack(
            "<IIIIfffff", p[:HDR])
        if magic != MAGIC:
            print(f"bad magic at frame {i}: 0x{magic:08x}", file=sys.stderr)
            return 2
        out = {
            "seq": seq,
            "version": ver,
            "flags": flags,
            "rnnoise_on": bool(flags & 1),
            "eq_on":      bool(flags & 2),
            "delay_on":   bool(flags & 4),
            "reverb_on":  bool(flags & 8),
            "vad_prob": round(vad, 4),
            "rms_in_db": round(rin, 2),
            "rms_out_db": round(rout, 2),
            "reduction_db": round(red, 2),
            "tracked_pitch_hz": round(pitch, 2),
        }
        if not args.skip_waveform:
            wi = struct.unpack(f"<{WAVEFORM_SAMPLES}f", p[HDR:HDR + WAV_BYTES])
            wo = struct.unpack(f"<{WAVEFORM_SAMPLES}f", p[HDR + WAV_BYTES:SPEC0])
            out["waveform_in_max"] = round(max(abs(x) for x in wi), 4)
            out["waveform_out_max"] = round(max(abs(x) for x in wo), 4)
        if not args.skip_spectrum:
            si = struct.unpack(f"<{SPECTRUM_BINS}f", p[SPEC0:SPEC0 + SPEC_BYTES])
            so = struct.unpack(f"<{SPECTRUM_BINS}f", p[SPEC0 + SPEC_BYTES:PKT])
            out["spectrum_in_min"] = round(min(si), 2)
            out["spectrum_in_max"] = round(max(si), 2)
            out["spectrum_out_min"] = round(min(so), 2)
            out["spectrum_out_max"] = round(max(so), 2)
        sys.stdout.write(json.dumps(out) + "\n")
        emitted += 1
        if args.limit and emitted >= args.limit:
            break
    if emitted == 0:
        print(f"no frames in {len(raw)} bytes", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
