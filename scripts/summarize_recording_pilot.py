"""Build a readable pilot report, plots, and a local listening/review page."""
import csv
import html
import json
import os
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pilot"
CONFIRMED_STATUSES = {"speaker_reviewed", "speaker_confirmed"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def stats(rows):
    duration = sum(float(r["audio_seconds"]) for r in rows)
    return dict(samples=len(rows), cer=sum(int(r["errors"]) for r in rows) /
                sum(int(r["reference_characters"]) for r in rows),
                rtf=sum(float(r["inference_seconds"]) for r in rows) / duration)


def main():
    manifest = read_csv(ROOT / "data/manifest.csv")
    references_reviewed = all(r.get("transcript_status") in CONFIRMED_STATUSES for r in manifest)
    noise_labels = {Path(r["path"]).stem: r["label"].replace("_", " ")
                    for r in read_csv(ROOT / "data/noise_manifest.csv")}
    audit = json.loads((ROOT / "outputs/recording_audit.json").read_text())
    all_rows, clean, noise = [], [], []
    for backend in ("whisper", "sensevoice"):
        clean.extend(read_csv(OUT / f"{backend}_clean.csv"))
        noise.extend(read_csv(OUT / f"{backend}_noise.csv"))
    all_rows = clean + noise
    summary = []
    for backend in ("whisper", "sensevoice"):
        for condition in sorted({r["condition"] for r in all_rows}):
            for method in ("off", "highpass", "spectral"):
                rows = [r for r in all_rows if r["backend"] == backend and r["condition"] == condition and r["denoise"] == method]
                summary.append(dict(backend=backend, condition=condition, denoise=method, **stats(rows)))
    write_csv(OUT / "summary.csv", summary)
    lines = ["# Recording pilot results", "",
             ("Status: small pilot. The speaker confirmed that the filenames contain the exact spoken words. "
              "The reference text and measured scores are unchanged; no recognition rerun was needed." if references_reviewed else
              "Status: provisional. At least one filename-derived reference transcript has not been confirmed by the speaker."), "",
             f"Prepared {audit['unique_speech_files']} unique speech recordings ({audit['unique_speech_seconds']:.2f} seconds), "
             "three noise recordings, and 63 synthetic noisy recordings. The Python/API recording appeared twice; only one copy is scored.", "",
             "Original files remain in `wav_docs/`. Prepared audio is mono, 16 kHz, PCM16. No gain adjustment or denoising was applied to the clean copies.", "",
             "## Clean recordings", "",
             "Both recognizers use automatic language detection. CER ignores punctuation, spacing, case, and full-width variants; "
             "it does not equate simplified/traditional Chinese, digits/written numbers, English/katakana spellings, or contractions. "
             "Some counted differences therefore reflect writing conventions. These are raw character error rates, not comprehension scores.", "",
             "| Recognizer | Group | Clips | Raw CER | RTF |", "|---|---|---:|---:|---:|"]
    for backend in ("whisper", "sensevoice"):
        for label, conditions in (("Mandarin incl. Python/API", {"clean_zh", "clean_zh+en"}),
                                  ("English", {"clean_en"}), ("Chinese + Japanese", {"clean_zh+ja"}),
                                  ("English + Japanese", {"clean_en+ja"})):
            s = stats([r for r in clean if r["backend"] == backend and r["denoise"] == "off" and r["condition"] in conditions])
            lines.append(f"| {backend} | {label} | {s['samples']} | {s['cer']:.1%} | {s['rtf']:.3f} |")
    lines += ["", "RTF is inference time / audio duration; below 1 means inference is faster than the recording duration. "
              "Loading and one warm-up are excluded, and preprocessing is timed separately. Single-run timings are preliminary.", "",
              "## Noise and denoising", "",
              "Each of seven unique clips was mixed with each noise at 20, 10, and 0 dB SNR. Every model/condition was run "
              "with denoising off, an 80 Hz high-pass, and spectral gating (strength 0.65), giving 420 scored recognitions including clean clips. "
              "SNR uses whole-clip RMS, and peaks are jointly scaled to avoid clipping. Noise starts at its beginning for each mix. "
              "This is a controlled pilot; synthetic mixtures do not replace real noisy-room recordings.", "",
              f"Noise labels: `keyboard` = {noise_labels['keyboard']}; `noise1` = {noise_labels['noise1']}; "
              f"`noise2` = {noise_labels['noise2']}. The speaker identified noise1 as normal ambient background "
              "and noise2 as room noise with the window open. The window position for noise1 was not specified. "
              "`noise1` is quiet (RMS about -62 dBFS), which is consistent with its role as an ambient baseline; "
              "low level alone is not a reason to discard or re-record it. SNR-controlled mixing amplifies its "
              "background and recording noise together and does not preserve the room's original noise level. "
              "`keyboard` and `noise2` have a few near-full-scale samples.", "",
              "The plot below covers the four Mandarin / Python-API clips only. All language groups and individual "
              "recognitions remain in the CSV files.", "", "![Noise experiment](noise_cer.png)", "",
              "| Recognizer | Method | Noisy clips, all languages | Raw CER | RTF |", "|---|---|---:|---:|---:|"]
    for backend in ("whisper", "sensevoice"):
        for method in ("off", "highpass", "spectral"):
            s = stats([r for r in noise if r["backend"] == backend and r["denoise"] == method])
            lines.append(f"| {backend} | {method} | {s['samples']} | {s['cer']:.1%} | {s['rtf']:.3f} |")
    lines += ["", "## Speech synthesis", "",
              "Both models synthesized the same six texts at configured speed 1.0, CPU, four threads. "
              "VITS uses speaker 3; Kokoro v1.1 uses Chinese voice 3 (zf_001), verified against the official version-specific voice list. "
              "Chinese normalization rules are enabled. The VITS output is 8 kHz and Kokoro is 24 kHz; "
              "these native output rates are preserved for listening. Voice identity and actual speaking rate also differ between models.", "",
              "| Model | Generated clips | RTF |", "|---|---:|---:|"]
    listening = []
    for backend in ("vits", "kokoro"):
        rows = read_csv(OUT / f"tts_{backend}/results.csv")
        rtf = sum(float(r["generation_seconds"]) for r in rows) / sum(float(r["audio_seconds"]) for r in rows)
        lines.append(f"| {backend} | {len(rows)} | {rtf:.3f} |")
        for row in rows:
            listening.append(dict(backend=backend, source=f"tts_{backend}/{row['file']}", text=row["text"]))
    random.Random(20260916).shuffle(listening)
    (OUT / "listening").mkdir(exist_ok=True)
    for i, row in enumerate(listening, 1):
        row["sample_id"] = f"S{i:02d}"
        row["file"] = f"listening/{row['sample_id']}.wav"
        shutil.copyfile(OUT / row["source"], OUT / row["file"])
    write_csv(OUT / "listening_key.csv", listening)
    if not (OUT / "listening_scores_template.csv").exists():
        write_csv(OUT / "listening_scores_template.csv", [dict(listener_id="", sample_id=r["sample_id"], naturalness="", clarity="", notes="") for r in listening])
    turns = json.loads((OUT / "conversation/turns.json").read_text(encoding="utf-8"))
    conversation_config = json.loads((OUT / "conversation/config.json").read_text(encoding="utf-8"))
    lines += ["", "No human quality ratings have been filled in. Open `review.html` to listen to randomized, anonymously labelled samples "
              "and export ratings. Keep `listening_key.csv` away from listeners until they finish. Three to five independent listeners "
              "are enough for an exploratory assessment, not a standardized MOS study.", "", "## Real conversation check", "",
              "The VITS lexicon reported omitted English words in the Python/API text; this is a documented model limitation to check in listening. "
              "Kokoro initially failed to phonemize those words with an explicit `zh` fallback. Using the model's default "
              "phonemizer resolved that error, and its six samples were regenerated. An unknown punctuation-token warning remains at Kokoro loading.", "",
              f"Ran {len(turns)} turns using SenseVoice ONNX → Ollama qwen2.5:3b → {conversation_config['tts']['backend']}. "
              "Four turns used your Mandarin recordings, followed by one text question referring to the conversation. "
              "Saved replies and stage timings in `conversation/`. This verifies actual inference and history delivery, "
              "but not live microphone capture, speaker playback, or whether every answer is correct. "
              "The LLM reply incorrectly promised a scheduled reminder, even after the system instruction was updated "
              "to state that reminders, cameras, and physical actions are unavailable. This remains an observed answer-quality failure: "
              "no reminder is actually scheduled. The prototype must not be presented as an action-execution system. "
              "The current app default is SenseVoice plus Kokoro, pending human listening review.", "",
              "## Reproduction", "", "From the project root:", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe scripts/run_recording_pilot.py",
              ".\\.venv\\Scripts\\python.exe scripts/summarize_recording_pilot.py", "```", "",
              "Use `config.local.toml`. Model URLs, revisions, downloaded-file SHA256 hashes, and sizes are in `models/sources.json`. "
              "Each benchmark JSON records configuration, package versions, and loading time. Whisper small uses CTranslate2 int8 on CPU "
              "with VAD; SenseVoice uses the int8 sherpa-onnx export on CPU, four threads, ITN enabled, without added VAD. "
              "This compares these configured systems, not an isolated neural-model architecture. RAM/VRAM peaks were not sampled.", "",
              "## Your remaining tasks", "",
              "The current filenames and noise descriptions have been confirmed by the speaker. No further transcript "
              "confirmation is needed for this batch. Recognition outputs do not override those reference words.", "",
              "1. For the original 30–50-clip Mandarin evaluation, you currently have four applicable unique clips; "
              "add 26–46 varied Mandarin recordings. Keep the mixed Japanese/English examples as a separately reported extension. "
              "Use these pilot clips for setup; reserve fresh recordings for the final test after settings are fixed.",
              "2. Record several sentences in a real noisy environment, test five live microphone exchanges, "
              "and ask 3–5 people to complete the listening ratings.",
              "3. Longer 30–60 second ambient and window-open noise recordings would be useful for the final dataset. "
              "The current quiet-room baseline remains usable for this pilot.",
              "4. After the final dataset and settings are fixed, repeat important timing comparisons three times, "
              "record memory/VRAM use, and make the short demonstration video. "
              "The broader vision/PDDL project described in total_aim.md is outside this speech pilot.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    make_plot(noise, references_reviewed, noise_labels)
    make_review(manifest, clean, listening, noise_labels)
    print(f"Created {OUT / 'report.md'} and {OUT / 'review.html'}")


def make_plot(rows, references_reviewed, noise_labels):
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, name in zip(axes, ("keyboard", "noise1", "noise2")):
        for backend, color in (("whisper", "#2563eb"), ("sensevoice", "#b45309")):
            for method, style in (("off", "-"), ("highpass", "--"), ("spectral", ":")):
                ys = []
                for snr in (20, 10, 0):
                    subset = [r for r in rows if r["backend"] == backend and r["denoise"] == method
                              and r["condition"] == f"{name}_{snr}dB" and Path(r["path"]).stem.split("_")[1] in {"001", "002", "003", "004"}]
                    ys.append(stats(subset)["cer"] * 100)
                ax.plot([20, 10, 0], ys, marker="o", color=color, linestyle=style, label=f"{backend} / {method}")
        ax.set_title(f"{name}: {noise_labels[name]}")
        ax.set_xlabel("SNR (dB; lower = more noise)")
        ax.set_xticks([20, 10, 0])
        ax.invert_xaxis()
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Raw character error rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    status = "speaker-confirmed transcripts" if references_reviewed else "filename transcripts unconfirmed"
    fig.suptitle(f"Pilot only: four Mandarin / Python-API clips; {status}")
    fig.tight_layout(rect=(0, .15, 1, .93))
    fig.savefig(OUT / "noise_cer.png", dpi=180)
    plt.close(fig)


def make_review(manifest, clean, listening, noise_labels):
    cards = []
    for i, row in enumerate(manifest):
        ident = Path(row["path"]).stem
        predictions = [r for r in clean if Path(r["path"]).stem == ident and r["denoise"] == "off"]
        details = "".join(f"<p><b>{html.escape(r['backend'])}</b>: {html.escape(r['hypothesis'])}</p>" for r in predictions)
        checked = " checked" if row.get("transcript_status") in CONFIRMED_STATUSES else ""
        cards.append(f'<article><h3>Recording {ident} · {html.escape(row["language"])}</h3>'
                     f'<audio controls preload="none" src="../../data/{row["path"]}"></audio>'
                     f'<label>Exact words spoken<textarea id="text-{i}">{html.escape(row["text"])}</textarea></label>'
                     f'<label><input type="checkbox" id="checked-{i}"{checked}> The speaker confirms these are the exact spoken words</label>{details}</article>')
    noise_html = ''.join(f'<article><h3>{name}: {html.escape(noise_labels[name])}</h3><audio controls preload="none" src="../../data/noise/{name}.wav"></audio></article>' for name in ('keyboard', 'noise1', 'noise2'))
    ratings = []
    for row in listening:
        sid = row["sample_id"]
        selects = ''.join(f'<label>{label}<select id="{sid}-{metric}"><option value="">Choose</option>' + ''.join(f'<option>{i}</option>' for i in range(1, 6)) + '</select></label>' for metric, label in (("naturalness", "Naturalness (1 poor – 5 excellent)"), ("clarity", "Clarity (1 poor – 5 excellent)")))
        ratings.append(f'<article><h3>{sid}</h3><p>{html.escape(row["text"])}</p><audio controls preload="none" src="{row["file"]}"></audio>{selects}<label>Pronunciation / missing words / comments<textarea id="{sid}-notes"></textarea></label></article>')
    page = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Speech pilot review</title>
<style>body{font:16px/1.55 system-ui,sans-serif;max-width:860px;margin:40px auto;padding:0 20px;color:#172033;background:#f5f7fb}h1,h2,h3{line-height:1.25}article{background:white;padding:20px;margin:16px 0;border:1px solid #dce2eb;border-radius:10px}audio{width:100%;margin:8px 0}label{display:block;margin:12px 0}textarea{box-sizing:border-box;width:100%;min-height:75px;padding:10px;font:inherit}button{background:#245dcc;color:white;padding:12px 18px;border:0;border-radius:6px;font:inherit;cursor:pointer}select,input{font:inherit;padding:4px}nav a{margin-right:16px}.note{background:#fff4d1;padding:14px;border-radius:8px}</style>
<h1>Speech pilot review</h1><p>Listen, check the transcripts, and rate the generated voices. Files stay on this computer; this page sends nothing.</p>
<nav><a href="#transcripts">Transcripts</a><a href="#noise">Noise</a><a href="#voices">Voice ratings</a></nav>
<p class="note">The speaker has confirmed the filename transcripts. Results are still a small pilot. Model outputs do not replace the reference words. Any new edits stay on this page until you download a CSV.</p>
<h2 id="transcripts">1. Check the seven unique recordings</h2><p>The Python/API clip was duplicated in the source folders and is included once. Keep exactly what you said, including unusual grammar.</p>
TRANSCRIPTS
<button onclick="saveManifest()">Download corrected manifest.csv</button><p>Replace data/manifest.csv in the project with this download, then rerun the experiments. Unchecked rows stay marked unconfirmed.</p>
<h2 id="noise">2. Noise recordings</h2><p>The speaker identified noise1 as normal ambient background and noise2 as room noise with the window open. The quiet ambient recording is a useful baseline; its low level alone does not require a replacement.</p>
NOISE
<h2 id="voices">3. Rate the generated voices</h2><p>Use headphones if possible. Rate every sample independently. Please do not inspect the answer key until you finish.</p><label>Listener ID <input id="listener" placeholder="e.g. listener01"></label>
RATINGS
<button onclick="saveRatings()">Download listening scores</button>
<script>
const manifest=MANIFEST;
const sampleIds=SAMPLE_IDS;
function download(name,rows){const keys=Object.keys(rows[0]);const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';const csv='\ufeff'+[keys,...rows.map(r=>keys.map(k=>r[k]))].map(r=>r.map(quote).join(',')).join('\r\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
function saveManifest(){const rows=manifest.map((r,i)=>({...r,text:document.getElementById('text-'+i).value.trim(),transcript_status:document.getElementById('checked-'+i).checked?'speaker_confirmed':'filename_unconfirmed'}));if(rows.some(r=>!r.text)){alert('Every recording needs a transcript.');return;}download('manifest.csv',rows);}
function saveRatings(){const listener=document.getElementById('listener').value.trim();if(!listener){alert('Please enter a listener ID.');return;}const rows=sampleIds.map(id=>({listener_id:listener,sample_id:id,naturalness:document.getElementById(id+'-naturalness').value,clarity:document.getElementById(id+'-clarity').value,notes:document.getElementById(id+'-notes').value}));download('listening_scores_'+listener.replace(/[^a-zA-Z0-9_-]/g,'_')+'.csv',rows);}
</script></html>'''
    page = page.replace("TRANSCRIPTS", "\n".join(cards)).replace("NOISE", noise_html).replace("RATINGS", "\n".join(ratings))
    page = page.replace("MANIFEST", json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c"))
    page = page.replace("SAMPLE_IDS", json.dumps([r["sample_id"] for r in listening]))
    (OUT / "review.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
