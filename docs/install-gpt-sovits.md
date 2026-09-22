# 从 GitHub 版本安装 GPT-SoVITS v4（Windows）

此仓库不包含 GPT-SoVITS 整合包或模型权重。本教程是在另一台 Windows 电脑上，从仓库代码配置 **GPT-SoVITS v4 基础声音**；RVC 音色转换是可选的后续步骤。GPT-SoVITS 用来把回答文字变成语音，Ollama 的 LLM 和 SenseVoice 的 ASR 仍分别运行。

本项目的服务启动脚本固定使用 `vitsModel/GPT-SoVITS-v2pro-20250604-nvidia50/` 目录、`runtime/python.exe` 和 `api_v2.py`。以下步骤面向有兼容 NVIDIA 显卡及驱动的 Windows 电脑。整合包来源、版本、许可和解压密码（若有）请查看提供者页面；不同日期、AMD、DirectML、Mac 或单独模型权重不能直接按此目录结构互换。下载文件较大，须预留足够磁盘空间。

## 1. 准备项目和基础依赖

在 PowerShell 中：

```powershell
git clone https://github.com/L9581/FinalYearLaiLai.git
cd FinalYearLaiLai
.\setup_new_computer.ps1
ollama pull qwen2.5:3b
```

先安装 Python 3.11/3.12、[Ollama](https://ollama.com/download/windows) 和 NVIDIA 驱动，并确认 `nvidia-smi` 能运行。`setup_new_computer.ps1` 只创建网页使用的 `.venv` 并安装 Python 依赖；它不下载整合包、模型权重或 Ollama。若尚未安装 Git，也可以从 GitHub 下载仓库 ZIP 并解压后在项目根目录执行后续命令。

从仓库克隆后还需下载当前 ASR 模型。可以运行以下脚本获取实验使用的 SenseVoice、Whisper、VITS 和 Kokoro 模型（会占额外空间）；如果只需要 SenseVoice，也可以仅从 `scripts/download_experiment_models.py` 指向的固定版本下载 `sensevoice-small-onnx/model.int8.onnx` 和 `tokens.txt` 到 `models/sensevoice-small-onnx/`：

```powershell
.\.venv\Scripts\python.exe scripts/download_experiment_models.py
```

## 2. 下载并摆放整合包

打开提供者的[整合包及模型下载链接](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/dkxgpiy9zb96hob4#KTvnO)。页面有多个版本和下载方式：优先选与本项目现用的 **2025-06-04 Windows NVIDIA 整合包**相符的版本。下载的是包含运行环境和 `api_v2.py` 的整合包，**不是**只下载单独的 `.pth` 文件。下载后按提供者说明解压，不要把压缩包当成可运行目录。

把解压出的实际程序根目录放到下面的位置；不同压缩包解压后的外层名字可能不同，以内部文件为准，不要多套一层目录：

```text
FinalYearLaiLai/
  vitsModel/
    GPT-SoVITS-v2pro-20250604-nvidia50/
      api_v2.py
      runtime/python.exe
      GPT_SoVITS/pretrained_models/
```

本项目选择的是 **v4**，请检查整合包内这些文件存在：

```text
GPT_SoVITS/pretrained_models/s1v3.ckpt
GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth
GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth
GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin
GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin
```

如果下载的整合包缺少 v4 权重或配套的中文模型，请从同一提供者页面补齐匹配版本，并按 `config.gpt-sovits-v4.yaml` 中的路径摆放；不要把 v2/v3 权重改名冒充 v4。接口文件名 `api_v2.py` 是上游脚本名，实际选择的版本由配置里的 `custom.version: v4` 决定。

## 3. 提供参考音频并配置

GPT-SoVITS 需要一段约 3–10 秒、清晰、单人说话的 WAV 以及**逐字准确的对应文本**。准备有权使用的声音，放到 `data/tts_reference/reference.wav`；这是模型的参考声音，不是训练集。创建目录并复制文件，例如：

```powershell
New-Item -ItemType Directory -Force data\tts_reference
Copy-Item -LiteralPath 'C:\path\to\your\reference.wav' -Destination data\tts_reference\reference.wav
Copy-Item config.voice-upgrade.example.toml config.voice-upgrade.toml
```

将 `C:\path\to\your\reference.wav` 换成实际路径。编辑新建的 `config.voice-upgrade.toml`：

```toml
[tts.gpt_sovits]
reference_audio = "data/tts_reference/reference.wav"
reference_text = "在这里填写这段 WAV 实际说出的完整文字。"
reference_language = "zh"

[tts.rvc]
enabled = false
```

上面只展示需要检查的字段，**不要用这段片段覆盖整个 TOML 文件**。模板中已默认关闭 RVC，首次安装先保持关闭；`pitch = 2` 只有启用 RVC 后才起作用。`config.voice-upgrade.toml`、参考音频及整合包都已被 `.gitignore` 排除，不会随 GitHub 仓库分发。

## 4. 启动与验证

在项目根目录运行：

```powershell
.\start_voice_lab.ps1
```

脚本会在 RVC 关闭时只启动 GPT-SoVITS 服务（`127.0.0.1:9880`）和网页（`127.0.0.1:8502`）；Ollama 服务需另外保持运行。打开 <http://localhost:8502>，先选“文字”，输入一句中文，确认有回答和可播放音频，再测麦克风。

也可运行检查命令；它检查路径和服务状态，**不等于实际合成成功**：

```powershell
.\.venv\Scripts\python.exe -m voice_lab.doctor --config config.voice-upgrade.toml --probe-ollama --probe-voice-services
```

如果打不开 `8502` 而进入 `8501`，通常是升级配置、整合包目录或 `nvidia-smi` 检测不满足，启动脚本回退到基础配置。查看终端提示，并核对 `vitsModel/GPT-SoVITS-v2pro-20250604-nvidia50/runtime/python.exe` 是否存在。第一次合成可能因模型加载耗时较长；服务日志位于 `outputs/voice_upgrade/gpt_sovits.stderr.log`。参考 WAV 路径或文本不匹配、v4 权重缺失、CUDA 驱动不兼容，都应先核对对应文件和日志。

## 可选：接入 RVC 音色

RVC 的 `.pth` 是**音色转换模型**，不能当成 GPT-SoVITS 的 v4 权重。要使用 RVC，先参考仓库中的 `scripts/download_rvc_runtime.py` 准备 RVC 代码、HuBERT 和 RMVPE，再将你有权使用的 RVC `.pth` 与配套 `.index` 放入 `rvc/`，把 `config.rvc.json` 中 `model`、`index` 路径改为实际位置，并设置 `config.voice-upgrade.toml` 的 `[tts.rvc].enabled = true`。本项目的 RVC 服务当前面向 **RVC v2、带音高、768 维索引**的模型，并默认用 CUDA；其他类型不能保证直接兼容。启动脚本此时会额外启动 `127.0.0.1:9881`。如果只想使用 v4 基础声音，可以一直关闭 RVC。

以上整合包下载页来自第三方，下载链接可能变动；本仓库不镜像或分发其权重。切换到另一平台或整合包版本时，先核对模型文件、运行环境和许可证，再调整脚本路径与配置。
