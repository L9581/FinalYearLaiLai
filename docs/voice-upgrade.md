# GPT-SoVITS v4 + 纳西妲 RVC

已在本机跑通：**SenseVoice → Ollama → GPT-SoVITS v4 → 纳西妲 RVC → 播放**。
新页面为 <http://localhost:8502>，配置文件是 `config.voice-upgrade.toml`。

## 现在怎么用

1. 打开新页面，侧栏“语音合成”选择 **GPT-SoVITS v4**。
2. 勾选“转换为纳西妲音色”，点击“试听当前声音”。
3. 音高默认 **+2 半音**，已按你的试听偏好保存。取消勾选可以试听 v4 的基础声音。
4. 选好后使用文字、上传 WAV 或麦克风对话即可。

另有[六组对照试听](../outputs/voice_upgrade/compare.html)，每组包含原来的 Kokoro、
v4 基础语音、v4 + 纳西妲三种输出。请在浏览器中打开这个 HTML 文件。
这些已有对照样音是在音高 **0** 时生成的；保存新默认值不会改变旧 WAV 和实验记录。
自动转写检查见 `outputs/voice_upgrade/pronunciation_check.csv`，不能代替人工听评。

## v4 实际用了什么

本地整合包已经包含 v4 所需文件，已使用独立的 `config.gpt-sovits-v4.yaml` 启动：

- `version: v4`，CUDA，半精度。
- GPT 权重：`GPT_SoVITS/pretrained_models/s1v3.ckpt`。
- SoVITS 权重：`GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth`。
- 声码器：`GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth`。

v4 使用 `s1v3.ckpt` 是该版本的配套配置；接口脚本名 `api_v2.py` 也不代表运行的是 v2。
服务日志已确认实际加载 **v4**，v4 输出为 **48 kHz**。

GPT-SoVITS 需要参考音频和它的准确文本。当前使用之前生成的合成女声
`data/tts_reference/kokoro_female.wav`，约 4.36 秒；没有使用你的个人录音来模仿你的声音。
以后若有更喜欢的干净讲话样音，可以换成 3–10 秒参考录音，并同步修改
`[tts.gpt_sovits].reference_audio` 和 `reference_text`。参考音频会影响音色和说话方式，
v4 版本号本身不保证一定更自然。

## RVC 配置

- 使用你提供的 `rvc/RVC_Nahida/nahida.pth` 和配套 `.index`。
- 模型实测为 **RVC v2、40 kHz、带音高**；索引维度为 768，共 32,787 个向量。
- RMVPE 提取音高，默认移调 **+2 半音**，索引混合比例 0.5，辅音保护 0.33。
- RVC 负责转换生成音频，不能把这个 `.pth` 作为 GPT-SoVITS 权重加载。
- RVC 输出保持原生 **40 kHz**。ASR 输入才会转换为 16 kHz。

本机服务使用整合包自带的 Python 3.9 / PyTorch 2.7 + CUDA 12.8；网页继续使用项目
`.venv` 中的 Python 3.12，两者通过本机 HTTP 连接。
服务只监听 `127.0.0.1:9880`（v4）和 `127.0.0.1:9881`（RVC）。
RVC 官方代码固定在提交 `81eed5e8f68b6bed1789f682fe78cdd324495afc`；模型下载来源和
SHA256 记录在 `models/rvc/sources.json`。主站连接超时时使用了 Hugging Face 镜像。

## 重启

从项目根目录运行 `start_voice_lab.ps1`，会先检查/启动语音服务，再启动新网页。
如果 PowerShell 不允许执行脚本，可以手动运行：

```powershell
.\.venv\Scripts\python.exe scripts/start_voice_services.py
$env:VOICE_LAB_CONFIG = "config.voice-upgrade.toml"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --server.headless true
```

Ollama 需保持运行，模型为 `qwen2.5:3b`。首次推理会有模型加载和计算内核初始化等待；
本次 v4 首次请求约 96 秒，RVC 首次请求约 18 秒，后续明显缩短。
启动日志和进程号保存在 `outputs/voice_upgrade/`。

## 已验证与局限

- 六条测试文本分别生成 v4、v4 + RVC，共 12 个新 WAV，包含数字、日期和中英混读。
- 预热后，本批 v4 生成耗时约 1.9–6.7 秒；RVC 后五条转换约 0.30–0.37 秒。
- 用你的 `004.wav` 跑通完整语音问答，输出 40 kHz WAV；该次总耗时约 5.9 秒。
  结果在 `outputs/voice_upgrade/full_pipeline.json` 和 `full_pipeline.wav`。
- 24 项自动测试通过；还实际点击了网页试听按钮，确认 RVC 开启输出 40 kHz，关闭输出 48 kHz。
- 自动转写仍对 “API” 有识别差异，不能据此保证英文缩写发音完全正确；请重点听第六组。
- 你已试听并选定 +2 半音为舒适的音高；最终演示仍需检查是否读错或失真。换音色不会自动修正上游读错的词。
- 提醒、摄像头和物理动作仍未实现；本次升级只涉及语音生成和转换。

原始模型对比仍使用 `config.local.toml`，原始试验结果保留在 `outputs/pilot/`。
新试听与结果在 `outputs/voice_upgrade/`。复现新样音：

```powershell
.\.venv\Scripts\python.exe scripts/compare_voice_upgrade.py
.\.venv\Scripts\python.exe -m voice_lab.doctor --config config.voice-upgrade.toml --probe-ollama --probe-voice-services
```
