# VeriTrust Audio

Audio media authenticity checking. Upload a file, get a three band verdict on whether it is a real recording or was generated, with the ensemble score shown.

On an audio file the system runs an ensemble of voice spoofing and synthetic speech detectors over overlapping windows. The verdict is `likely_authentic`, `uncertain`, or `likely_ai_generated`. The middle band is a deliberate abstention rather than a rounding artifact.

## The audio pathway

**Seven checkpoints across three architecture families.** Five wav2vec2 or XLS-R waveform models, one WavLM, and one Audio Spectrogram Transformer that reads a mel spectrogram as a patch grid. The AST carries a weight of 1.2 rather than 1.0, which is the only non uniform weight here and is a prior rather than a measurement: an ensemble of one architecture largely agrees with itself.

**Overlapping windows, not one score for the whole file.** Four second windows on a two second hop. A cloned sentence spliced into an otherwise real recording is a small fraction of a three minute file, and one score for the whole clip averages it into nothing. Overlap matters because a splice that lands on a window boundary is split across two windows and weakened in both.

**Windows combine at the 0.9 quantile, deliberately not at the max.** A max over N windows only ever climbs as N grows, so a longer recording would score higher for being longer, which is a length detector wearing a spoofing detector's clothes. The quantile interpolates, so it sits below the loudest window and does not drift upward with duration.

**Silent windows are skipped and said to be skipped.** A window below the RMS floor carries no voice to judge. If every window is silent the loudest one is kept anyway with a note saying it should not be trusted.

**Resampling refuses rather than degrades.** These checkpoints expect 16 kHz. Naive decimation aliases high frequency content down into exactly the band the spoofing detectors read, manufacturing the artifacts they are looking for, so a request is rejected with the reason named when no anti-aliasing resampler is available.

## Known limits

Lossy compression, phone codecs and re-encoding strip exactly the fine detail spoofing detectors read, so a voice note forwarded through two messaging apps is a harder case than the same audio as a WAV. Background music and overlapping speakers are outside what these checkpoints were trained on. Detection is strongest against the TTS and voice cloning systems each model saw, so a newer synthesiser is detected less reliably.

## Quickstart

Python 3.10 or newer.

```bash
git clone https://github.com/FaraazKhatib/audio-and-image-partial-.git
cd audio-and-image-partial-
setup.bat            # Windows
./setup.sh           # macOS or Linux
```

Then start it:

```bash
start.bat            # Windows
./start.sh           # macOS or Linux
```

Open http://localhost:8000.

## Configuration

Every setting is an environment variable. The ones most worth knowing:

| Variable | Default | Effect |
| --- | --- | --- |
| `VT_DEVICE` | `auto` | `cuda`, `mps`, `cpu` |
| `VT_AUTHENTIC_MAX` | `0.35` | Upper edge of the authentic band |
| `VT_AI_MIN` | `0.65` | Lower edge of the AI band |
| `VT_MAX_CONCURRENCY` | `2` | Simultaneous inferences. Raise only if you have VRAM to spare |
| `VT_SIGNAL_CLAMP` | `0.02` | Bound on each model reading before its logit is taken |
| `VT_ENABLE_AUDIO` | `true` | Off refuses audio uploads with a note saying so |
| `VT_AUDIO_WINDOW_SECONDS` | `4.0` | Window length fed to each audio checkpoint |
| `VT_AUDIO_HOP_SECONDS` | `2.0` | Window step. Overlap is what catches a splice on a boundary |
| `VT_AUDIO_MAX_WINDOWS` | `24` | Cap per file, sampled across the whole clip |
| `VT_AUDIO_QUANTILE` | `0.9` | How windows combine. |

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/health` | Liveness plus whether any detector is ready |
| GET | `/api/v1/models` | Loaded detectors, failures with reasons, device, thresholds |
| POST | `/api/v1/analyze` | Multipart upload, audio modality read from the file's bytes |
| POST | `/api/v1/analyze-base64` | JSON body with base64 content, same sniffing |
