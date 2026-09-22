# 验证记录

## 2026-09-16：真实录音试验

- 处理 `wav_docs/` 中的 8 个语音文件和 3 个噪声文件；排除一份重复录音后，保留 7 条独立语音，共 50.22 秒。
- 文件名生成参考文本，转为 16kHz 单声道 PCM16 副本，原文件未改动。说话人随后确认文件名与实际说话内容完全一致；noise1 是日常环境底噪，noise2 是房间开窗时的噪声。已更新确认状态和标签，音频、参考文本、识别结果均未改变，无需重跑识别。
- Whisper small / CTranslate2 int8 与 SenseVoice / sherpa-onnx int8 在 CPU 上完成 420 条识别结果：7 条原始语音、63 条噪声混音，分别测试关闭降噪、高通、谱门控。
- VITS 与 Kokoro 各生成 6 条真实 TTS 音频。VITS 原生输出为 8kHz，Kokoro 为 24kHz。修正 Kokoro 的语言回退配置后，Python/API 的音素转换错误消失；VITS 仍报告忽略英文词。
- 使用 SenseVoice → Ollama qwen2.5:3b → Kokoro 完成 4 条真实录音请求和 1 条文字上下文追问，保存 5 个回答 WAV、转写、回答和耗时。
- 语言模型仍会错误承诺稍后提醒，即使系统指令明确说明无此能力。这是保留的失败案例，程序没有实际设置提醒。
- `python -m pytest -q --basetemp outputs/pytest-recording-pilot`：**20 passed**。另检查了 420 条结果数值、17 个生成 WAV、复核页面的 JavaScript 语法/CSV 导出和 22 个音频链接。
- `doctor --config config.local.toml --probe-ollama` 通过；Streamlit 本机服务健康检查返回 HTTP 200。

结果与图表：[试验报告](../outputs/pilot/report.md)。人工复核：[浏览器页面](../outputs/pilot/review.html)。复现与待办：[录音工作流](recording-workflow.md)。

仍未验证：真实麦克风和扬声器现场使用、多人听评、校对后的最终测试集、真实噪声环境、峰值内存/显存、重复测量。当前结果是小样本初步实验，不是最终结论。

## 2026-09-14：第一版自动测试

验证日期：2026-09-14。系统：Windows，Python 3.12.8。项目依赖安装于 `.venv`，全局 Python 未修改。安装使用清华 PyPI 镜像；SenseVoice 可选依赖未安装。

## 已验证

执行 `.\.venv\Scripts\python.exe -m pytest -q`：**20 passed**。

- PCM WAV 解码、立体声转单声道、16kHz 重采样、空或无效音频拒绝。
- 高通滤波对低频信号的抑制；真实 noisereduce 对短录音的处理。
- 已知信噪比混音、中文 CER、按字符总数汇总结果及按噪声条件分组。
- 用替代 ASR / LLM / TTS 检查整条流程、对话历史、输出 WAV、错误处理。
- 用 Streamlit AppTest 检查文字发送、刷新不重复发送、清空对话和输入。
- 实际导入 faster-whisper；用已安装 sherpa-onnx 检查 VITS / Kokoro 的构造参数，核对 `generate(text, sid, speed)` 接口。
- 核对 sherpa-onnx 官方模型发布页和 Python 示例；确认 README 中两个 TTS 包的下载名称。

## 尚未验证

- 没有下载 ASR、VITS、Kokoro 或 Ollama 模型权重。
- 没有运行真实 ASR → LLM → TTS 端到端对话。
- 没有进行实际麦克风录音、扬声器播放或 GPU 推理验证。
- SenseVoice 接入口已编写，未安装其依赖或执行真实推理。
- 没有识别准确率、真实模型速度、音质或真实噪声环境结论。

自动测试中的替代模型只验证程序流程，不是模型评测数据。接下来按 README 下载模型并完成首次真实对话，再按两周计划采集实验数据。
