# Working with your recordings

The current batch is in `wav_docs/`. The speaker confirmed that its filenames
contain exactly what was said; the transcripts are marked `speaker_confirmed`.
`noise1` is normal ambient room background, and `noise2` is room noise with the
window open. The window position for `noise1` was not specified. Confirmation is
recorded in `data/recording_confirmation.json`.
The Python/API sentence is duplicated across folders; the initial pilot has seven
unique speech recordings and three noise recordings. Four of the unique speech
recordings cover the Mandarin baseline, including the Python/API example.

## Review the results

- Open `outputs/pilot/review.html` in a browser for audio playback, transcript
  corrections, and anonymous TTS listening ratings. Opening this file works locally.
- Read `outputs/pilot/report.md` for the measured results and limitations.
- `data/transcript_review.csv` maps the source filenames to the prepared recordings
  and records the duplicate. `outputs/recording_audit.csv` records format, levels,
  durations, and duplicate hashes.
- `data/manifest.csv` is the authoritative input for rerunning the pilot. Its
  filename-derived reference words are confirmed by the speaker; automatic
  recognizer outputs do not replace them.

No further transcript confirmation is needed for this batch. Scores are unchanged
because confirmation did not change the reference words or audio. The review page
still permits later corrections and CSV export. Its checkboxes show speaker
confirmation. The Mandarin and source-review CSVs have also been updated; future
manual replacements of the main manifest do not automatically update those copies.

## Run the app

In PowerShell from the project root:

```powershell
$env:VOICE_LAB_CONFIG = "config.local.toml"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.headless true
```

Alternatively run `start_voice_lab.ps1` if local PowerShell script execution is
enabled. Open <http://localhost:8501>. Select an ASR and TTS in the sidebar, upload
a WAV or use the microphone, and send it. The local configuration enables automatic
language detection; both TTS models here are intended for the Chinese responses.
Speaker 3 is valid for the configured VITS model and is the Chinese `zf_001` voice
in this exact Kokoro v1.1 package.

The local app defaults to SenseVoice and Kokoro for this pilot. The model still
falsely promised a future reminder in testing despite the capability instruction.
No reminder, camera access, or physical action is implemented; treat that response
as a recorded failure case for the broader project.

SenseVoice uses the local int8 ONNX export with the already installed sherpa-onnx
runtime. Set `sensevoice_runtime = "sherpa-onnx"`, along with the local model and
token paths, to use it; the existing FunASR route remains the default when that
setting is absent. This avoids needing another Python/PyTorch environment for
these experiments. The two runtime/export versions should not be treated as
identical benchmarks.

## Repeat the pilot after correcting transcripts

```powershell
.\.venv\Scripts\python.exe scripts/run_recording_pilot.py
.\.venv\Scripts\python.exe scripts/summarize_recording_pilot.py
```

The runner generates the noise variants, runs both recognizers with all three
denoising settings, generates both sets of TTS samples, and tests four recorded
Chinese requests plus a text follow-up through the real conversation pipeline.
Ollama must be running with `qwen2.5:3b`. Output files for this pilot are overwritten
when rerun. Save any listening scores outside the generated output filenames.

The initial preparation script, `scripts/prepare_recordings.py`, is specifically
for importing the current filename-labelled folder structure. Do not rerun it
after manually correcting the manifest: it regenerates the provisional labels.
Original recordings are never changed by the script. For more recordings, first
preserve reviewed transcripts, then import/merge the new entries.

`scripts/download_experiment_models.py` documents the model setup. ASR revisions
are pinned; TTS packages come from official sherpa-onnx releases. Sources and hashes
are recorded in `models/sources.json`; weights stay inside this workspace.

## Remaining human work

1. For the planned 30–50-item Mandarin evaluation, add 26–46 distinct Mandarin
   utterances. Keep mixed Japanese/English examples as a separate exploratory
   group. Reserve fresh recordings for final testing after settings are fixed.
2. Record some speech in a real noisy location, test five live microphone exchanges,
   and collect 3–5 independent listeners' ratings using the review page.
3. Aim for 30–60 seconds per final noise sample. The current quiet ambient clip is
   a valid baseline; its low level alone is not a reason to replace it. SNR-controlled
   mixing changes its level and does not reproduce the original room noise level.
4. Repeat important final timing runs three times, record memory/VRAM use, and make
   the 2–3 minute demonstration. The pilot report is a starting point for the final
   write-up, not a completed final evaluation.
