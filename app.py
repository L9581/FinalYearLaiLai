import json
import os
from pathlib import Path

import streamlit as st

from voice_lab.config import load_config
from voice_lab.audio import wav_bytes
from voice_lab.pipeline import VoicePipeline


@st.cache_resource(max_entries=1)
def get_pipeline(serialized_config: str) -> VoicePipeline:
    return VoicePipeline(json.loads(serialized_config))


def main():
    st.set_page_config(page_title="本地语音交互实验室", page_icon="🎙️", layout="centered")
    st.title("本地语音交互实验室")
    st.caption("录一段话，听模型回答。也可以上传 WAV 或输入文字。")
    default_config = Path(__file__).with_name("config.toml")
    try:
        config = load_config(os.environ.get("VOICE_LAB_CONFIG", str(default_config)))
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"配置读取失败：{exc}")
        st.stop()

    with st.sidebar:
        st.header("实验设置")
        asr_names = {"whisper": "Whisper（faster-whisper）", "sensevoice": "SenseVoiceSmall"}
        asr_options = list(asr_names)
        config["asr"]["backend"] = st.selectbox(
            "语音识别", asr_options, index=asr_options.index(config["asr"]["backend"]),
            format_func=asr_names.get,
        )
        methods = {"off": "关闭", "highpass": "高通滤波（低频噪声）", "spectral": "频谱降噪"}
        config["audio"]["denoise"] = st.selectbox(
            "降噪", list(methods), index=list(methods).index(config["audio"]["denoise"]),
            format_func=methods.get,
        )
        tts_options = ["vits", "kokoro"]
        if "gpt_sovits" in config["tts"]:
            tts_options.append("gpt_sovits")
        config["tts"]["backend"] = st.selectbox(
            "语音合成", tts_options, index=tts_options.index(config["tts"]["backend"]),
            format_func=lambda name: {"vits": "VITS", "kokoro": "Kokoro", "gpt_sovits": "GPT-SoVITS v4"}[name],
        )
        if config["tts"]["backend"] != "gpt_sovits":
            config["tts"]["speaker_id"] = st.number_input(
                "音色编号", min_value=0, value=config["tts"]["speaker_id"], step=1,
                help="音色编号依具体模型而定，请参考所下载模型的说明。",
            )
        else:
            st.caption("基础声音由参考音频决定。开启下方音色转换可使用你选的声音。")
        config["tts"]["speed"] = st.slider("语速", .7, 1.3, float(config["tts"]["speed"]), .05)
        if "rvc" in config["tts"]:
            rvc = config["tts"]["rvc"]
            rvc["enabled"] = st.checkbox(f"转换为{rvc['voice_name']}音色", value=rvc["enabled"])
            if rvc["enabled"]:
                rvc["pitch"] = st.slider("音高调整（半音）", -12, 12, int(rvc["pitch"]),
                                        help="先保持 0；听起来过低或过高时再调整。")
        preview_clicked = st.button("试听当前声音")
        serialized = json.dumps(config, ensure_ascii=False, sort_keys=True)
        if preview_clicked:
            with st.spinner("正在生成试听语音，首次加载需要更长时间"):
                try:
                    pipeline = get_pipeline(serialized)
                    with pipeline.lock:
                        preview = pipeline.tts.synthesize("你好，很高兴和你聊天。我们可以一起讨论今天的学习计划。")
                    st.session_state.voice_preview = (serialized, wav_bytes(preview))
                except Exception as exc:
                    st.session_state.pop("voice_preview", None)
                    st.error(str(exc))
        saved_preview = st.session_state.get("voice_preview")
        if saved_preview and saved_preview[0] == serialized:
            st.audio(saved_preview[1], format="audio/wav", autoplay=preview_clicked)
        st.caption(f"语言模型：{config['llm']['model']}")
        st.caption("使用前需安装依赖、准备 TTS 权重并启动 Ollama。详见项目 README。")
        if st.button("清空对话"):
            st.session_state.history = []
            st.session_state.pop("last_result", None)
            st.session_state.input_generation = st.session_state.get("input_generation", 0) + 1

    if "history" not in st.session_state:
        st.session_state.history = []
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    generation = st.session_state.get("input_generation", 0)
    mode = st.radio("输入方式", ["麦克风", "上传 WAV", "文字"], horizontal=True)
    audio = None
    text = ""
    if mode == "麦克风":
        audio = st.audio_input("点击录音，结束后发送", key=f"mic_{generation}")
        st.caption(f"每段最多 {config['audio']['max_seconds']} 秒；请在 localhost 页面允许麦克风访问。")
    elif mode == "上传 WAV":
        audio = st.file_uploader("选择 PCM WAV 文件（最多 30MB）", type=["wav"], key=f"file_{generation}")
    else:
        text = st.text_area("输入你想说的话", max_chars=4000, key=f"text_{generation}")

    submitted = st.button("发送并生成语音", type="primary", disabled=audio is None and not text.strip())
    if submitted:
        # A failed new turn must not display audio from the previous answer.
        st.session_state.pop("last_result", None)
        with st.status("正在准备模型", expanded=True) as status:
            try:
                pipeline = get_pipeline(json.dumps(config, ensure_ascii=False, sort_keys=True))
                result = pipeline.run(
                    audio_wav=audio.getvalue() if audio is not None else None,
                    text=text, history=st.session_state.history,
                    progress=lambda message: status.update(label=message),
                )
            except Exception as exc:
                status.update(label="本轮处理失败", state="error")
                st.error(str(exc))
                st.stop()
            st.session_state.history.extend([
                {"role": "user", "content": result.transcript},
                {"role": "assistant", "content": result.answer},
            ])
            st.session_state.history = st.session_state.history[-40:]
            st.session_state.last_result = result
            status.update(label="回答已生成" if not result.tts_error else "已生成文字，语音合成失败", state="complete")
        with st.chat_message("user"):
            st.write(result.transcript)
        with st.chat_message("assistant"):
            st.write(result.answer)

    result = st.session_state.get("last_result")
    if result is not None:
        if result.audio_wav:
            st.audio(result.audio_wav, format="audio/wav", autoplay=submitted)
            st.download_button("下载回答语音", result.audio_wav, "reply.wav", "audio/wav")
            st.caption("浏览器若阻止自动播放，请点击播放器。")
        if result.tts_error:
            st.warning(f"语音合成失败：{result.tts_error}")
        with st.expander("查看实验信息"):
            labels = {"preprocess": "预处理", "asr": "语音识别", "llm": "模型回答", "tts": "语音合成", "total": "总计"}
            rows = [f"| {labels[key]} | {value:.3f} |" for key, value in result.timings.items()]
            st.markdown("| 阶段 | 耗时（秒） |\n| --- | ---: |\n" + "\n".join(rows))
            st.caption("首次请求包含模型加载／可能的 ASR 下载时间；这里的总耗时从点击发送后计算，不含录音和播放。")
            if result.processed_wav:
                st.write("实际送入识别模型的音频")
                st.audio(result.processed_wav, format="audio/wav")


if __name__ == "__main__":
    main()
