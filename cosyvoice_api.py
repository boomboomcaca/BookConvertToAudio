"""CosyVoice WebUI API (headless, port 8000).

仅保留官方 CosyVoice TTS 推理接口, 供 pyvideotrans / 配音链路调用。
原 book-converter 有声书 GUI (7860) 已移除。
"""
import gradio as gradio_module
import inspect
import os
import sys
import time
import threading
import traceback
from typing import Any, cast

import torch  # type: ignore[import]
import torchaudio  # type: ignore[import]

gr = cast(Any, gradio_module)

# 设置环境路径
current_dir = os.getcwd()
cosyvoice_root = os.path.join(current_dir, 'CosyVoice')
sys.path.append(cosyvoice_root)
sys.path.append(os.path.join(cosyvoice_root, 'third_party', 'Matcha-TTS'))
# 如果有 ffmpeg, 添加路径
ffmpeg_path = os.path.join(cosyvoice_root, 'ffmpeg', 'bin')
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

# 禁用 DeepSpeed 检查
os.environ["DS_SKIP_CUDA_CHECK"] = "1"
os.environ["DS_BUILD_OPS"] = "0"

# 全局模型变量
cosyvoice_model: Any = None
# 当前模型版本标识: 'v3' / 'v2' / 'v1'
cosyvoice_model_version: str = ''

# 是否优先加载 RL 微调过的 LLM 权重 (llm.rl.pt): 质量优于 base llm.pt
USE_RL_LLM_IF_AVAILABLE = True

# 是否启用 TensorRT 加速 flow.decoder (DiT)
USE_TRT_IF_AVAILABLE = True

# 优先使用的模型目录 (按顺序尝试), CosyVoice3 优先
COSYVOICE_MODEL_DIR_CANDIDATES = [
    'Fun-CosyVoice3-0.5B',
    'CosyVoice2-0.5B',
    'CosyVoice-300M',
    'CosyVoice-300M-SFT',
    'CosyVoice-300M-Instruct',
]

# CosyVoice 模型推理锁, 防止并发推理导致 CUDA/模型状态损坏
cosyvoice_inference_lock = threading.Lock()


def locked_generator(gen, lock):
    # Cross-process GPU serialization: take the shared gpu_lock so book-converter
    # inference never runs concurrently with stash's ASR/dub/translate (all hold
    # the same lock) and OOM the shared RTX 3060. Per-block hold (stream=False),
    # released between blocks so the pipeline can interleave. Degrades to a no-op
    # if the lock module is unavailable.
    try:
        from gpu_lock import gpu_lock as _gpu_lock
        _ctx = _gpu_lock("book-converter")
    except Exception:
        import contextlib
        _ctx = contextlib.nullcontext()
    with _ctx:
        lock.acquire()
        try:
            for val in gen:
                yield val
        finally:
            lock.release()


def _resolve_cosyvoice_model_dir() -> str:
    """按优先级查找已下载的 CosyVoice 模型目录，优先返回 CosyVoice3。"""
    base_dir = os.path.join(cosyvoice_root, 'pretrained_models')
    for name in COSYVOICE_MODEL_DIR_CANDIDATES:
        candidate = os.path.join(base_dir, name)
        # 任一版本的 yaml 存在即视为有效模型目录
        for yaml_name in ('cosyvoice3.yaml', 'cosyvoice2.yaml', 'cosyvoice.yaml'):
            if os.path.exists(os.path.join(candidate, yaml_name)):
                return candidate
    # 兜底：返回 CosyVoice3 默认目录（即使不存在，也让后续报错更明确）
    return os.path.join(base_dir, COSYVOICE_MODEL_DIR_CANDIDATES[0])


def _try_load_rl_llm(model_dir: str) -> bool:
    """若存在 llm.rl.pt 且开启开关，则将其权重覆盖到已初始化的 cosyvoice_model.model.llm 上。

    返回是否成功加载 RL 权重。
    """
    global cosyvoice_model
    if not USE_RL_LLM_IF_AVAILABLE or cosyvoice_model is None:
        return False
    rl_path = os.path.join(model_dir, 'llm.rl.pt')
    if not os.path.exists(rl_path):
        return False
    try:
        inner = getattr(cosyvoice_model, 'model', None)
        llm_module = getattr(inner, 'llm', None) if inner is not None else None
        device = getattr(inner, 'device', None) if inner is not None else None
        if llm_module is None:
            print('[RL-LLM] cosyvoice_model.model.llm 不存在，跳过 RL 权重加载')
            return False
        print(f'[RL-LLM] Loading RL fine-tuned LLM weights from {rl_path}...')
        state_dict = torch.load(rl_path, map_location=device or 'cpu', weights_only=True)
        llm_module.load_state_dict(state_dict, strict=True)
        if device is not None:
            llm_module.to(device)
        llm_module.eval()
        print('[RL-LLM] RL weights loaded successfully.')
        return True
    except Exception as e:
        print(f'[RL-LLM] Failed to load RL weights, falling back to base LLM: {e}')
        return False


def _detect_cosyvoice_class(model_dir: str):
    """根据 model_dir 中的 yaml 文件选择匹配的 CosyVoice 类与版本号。"""
    if os.path.exists(os.path.join(model_dir, 'cosyvoice3.yaml')):
        from cosyvoice.cli.cosyvoice import CosyVoice3 as CosyVoiceCls
        return CosyVoiceCls, 'v3', 'CosyVoice3'
    if os.path.exists(os.path.join(model_dir, 'cosyvoice2.yaml')):
        from cosyvoice.cli.cosyvoice import CosyVoice2 as CosyVoiceCls
        return CosyVoiceCls, 'v2', 'CosyVoice2'
    if os.path.exists(os.path.join(model_dir, 'cosyvoice.yaml')):
        from cosyvoice.cli.cosyvoice import CosyVoice as CosyVoiceCls
        return CosyVoiceCls, 'v1', 'CosyVoice'
    raise FileNotFoundError(f"No valid cosyvoice*.yaml found in {model_dir}")


def load_model():
    global cosyvoice_model, cosyvoice_model_version
    if cosyvoice_model is None:
        try:
            model_dir = _resolve_cosyvoice_model_dir()
            CosyVoiceCls, version, class_name = _detect_cosyvoice_class(model_dir)
            print(f"Detected {class_name} class.")
            print(f"Loading model from {model_dir}...")

            def _supports_parameter(param_name: str) -> bool:
                """检查 CosyVoice 构造函数是否支持给定参数"""
                try:
                    return param_name in inspect.signature(CosyVoiceCls).parameters
                except (TypeError, ValueError):
                    return False

            def _build_kwargs(fp16: bool, load_trt: bool) -> dict[str, Any]:
                # CosyVoice3 不再支持 load_jit；仅在签名中存在时才传入对应参数
                kwargs: dict[str, Any] = {"fp16": fp16}
                for key in ("load_jit", "load_vllm"):
                    if _supports_parameter(key):
                        kwargs[key] = False
                if _supports_parameter("load_trt"):
                    kwargs["load_trt"] = load_trt
                return kwargs

            precision = ''
            trt_enabled = USE_TRT_IF_AVAILABLE and _supports_parameter("load_trt")
            try:
                if trt_enabled:
                    # FP16 + TRT：Tensor Core 加速 + 算子融合，实测 RTF 最优
                    print("Enabling TensorRT (FP16 for max throughput on Tensor Core GPUs)...")
                    print("First-time engine compilation takes ~30 seconds; subsequent boots reuse the cached .plan file.")
                    cosyvoice_model = CosyVoiceCls(model_dir, **_build_kwargs(True, True))
                    cosyvoice_model_version = version
                    precision = 'FP16+TRT'
                else:
                    cosyvoice_model = CosyVoiceCls(model_dir, **_build_kwargs(True, False))
                    cosyvoice_model_version = version
                    precision = 'FP16'
            except Exception as e:
                print(f"Initial load failed: {e}, falling back to FP32 (no TRT)...")
                cosyvoice_model = CosyVoiceCls(model_dir, **_build_kwargs(False, False))
                cosyvoice_model_version = version
                precision = 'FP32'

            # 加载完成后尝试用 RL 微调后的 LLM 权重覆盖，明显提升内容准确率
            rl_loaded = _try_load_rl_llm(model_dir)
            llm_tag = 'RL-LLM' if rl_loaded else 'base-LLM'
            return f"{class_name} loaded successfully ({precision}, {llm_tag})."
        except Exception as e:
            return f"Error loading model: {str(e)}"
    return "Model already loaded."


def _build_cosyvoice_api_demo(cosyvoice):
    """构建一个最小化的 Gradio Blocks,仅作为 pyvideotrans 等客户端的 API 通道。

    - UI 装饰(Markdown / Row / Column / 标签提示)全部去掉
    - 但保留 input/output 组件的类型与顺序,**严格匹配官方 CosyVoice/webui.py 的 generate_button.click 签名**
    - seed_button.click 仍注册在 generate_button.click 之前,保证 generate_audio 的 fn_index 与官方一致(=1)
    - 内置 CosyVoice3 必需的 `<|endofprompt|>` 注入及 `inference_instruct2` 切换
    """
    import random as _random
    import numpy as _np
    from cosyvoice.utils.file_utils import logging as _logging  # type: ignore[import]
    from cosyvoice.utils.common import set_all_random_seed as _set_seed  # type: ignore[import]

    inference_mode_list = ['3s极速复刻', '预训练音色', '跨语种复刻', '自然语言控制']
    stream_mode_list = [('否', False), ('是', True)]
    prompt_sr = 16000
    sft_spk = cosyvoice.list_available_spks() or ['']
    default_data = _np.zeros(cosyvoice.sample_rate)

    is_cv3 = cosyvoice.__class__.__name__ == 'CosyVoice3'
    eop_prefix = 'You are a helpful assistant.<|endofprompt|>'

    def _generate_seed():
        return {"__type__": "update", "value": _random.randint(1, 100000000)}

    def _generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text,
                        prompt_wav_upload, prompt_wav_record, instruct_text,
                        seed, stream, speed):
        if prompt_wav_upload is not None:
            prompt_wav = prompt_wav_upload
        elif prompt_wav_record is not None:
            prompt_wav = prompt_wav_record
        else:
            prompt_wav = None

        if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
            if prompt_wav is None:
                gr.Warning('prompt 音频为空，请提供 prompt 音频')
                yield (cosyvoice.sample_rate, default_data)
                return
            try:
                # cast(Any, torchaudio).info 等价于 torchaudio.info,运行时无差异;
                # 仅用于绕过 torchaudio 类型 stub 不全导致的 Pyright 误报
                actual_sr = cast(Any, torchaudio).info(prompt_wav).sample_rate
                if actual_sr < prompt_sr:
                    gr.Warning(f'prompt 音频采样率 {actual_sr} 低于 {prompt_sr}')
                    yield (cosyvoice.sample_rate, default_data)
                    return
            except Exception:
                pass

        if mode_checkbox_group == '预训练音色':
            if sft_dropdown == '':
                gr.Warning('没有可用的预训练音色')
                yield (cosyvoice.sample_rate, default_data)
                return
            _logging.info('get sft inference request')
            _set_seed(seed)
            for i in locked_generator(cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed), cosyvoice_inference_lock):
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
        elif mode_checkbox_group == '3s极速复刻':
            if prompt_text == '':
                gr.Warning('prompt 文本为空')
                yield (cosyvoice.sample_rate, default_data)
                return
            _logging.info('get zero_shot inference request')
            _set_seed(seed)
            if is_cv3 and '<|endofprompt|>' not in prompt_text:
                prompt_text = eop_prefix + prompt_text
            for i in locked_generator(cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed), cosyvoice_inference_lock):
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
        elif mode_checkbox_group == '跨语种复刻':
            _logging.info('get cross_lingual inference request')
            _set_seed(seed)
            if is_cv3 and '<|endofprompt|>' not in tts_text:
                tts_text = eop_prefix + tts_text
            for i in locked_generator(cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed), cosyvoice_inference_lock):
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
        else:
            if instruct_text == '':
                gr.Warning('请输入 instruct 文本')
                yield (cosyvoice.sample_rate, default_data)
                return
            _logging.info('get instruct inference request')
            _set_seed(seed)
            if is_cv3:
                it = instruct_text if '<|endofprompt|>' in instruct_text else (instruct_text + '<|endofprompt|>')
                for i in locked_generator(cosyvoice.inference_instruct2(tts_text, it, prompt_wav, stream=stream, speed=speed), cosyvoice_inference_lock):
                    yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
            else:
                for i in locked_generator(cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed), cosyvoice_inference_lock):
                    yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())

    # 极简 Blocks:只保留 API 必需的组件,且顺序/类型严格对齐官方 webui.py
    with gr.Blocks(title="CosyVoice API (headless)") as demo:
        tts_text = gr.Textbox(label="tts_text", value="")
        mode_checkbox_group = gr.Radio(choices=inference_mode_list, value=inference_mode_list[0], label="mode")
        sft_dropdown = gr.Dropdown(choices=sft_spk, value=sft_spk[0], label="sft_dropdown")
        stream_radio = gr.Radio(choices=stream_mode_list, value=False, label="stream")
        speed = gr.Number(value=1.0, label="speed", minimum=0.5, maximum=2.0, step=0.1)
        seed_button = gr.Button(value="\U0001F3B2")
        seed = gr.Number(value=0, label="seed")
        prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label="prompt_wav_upload")
        prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label="prompt_wav_record")
        prompt_text = gr.Textbox(label="prompt_text", value="")
        instruct_text = gr.Textbox(label="instruct_text", value="")
        generate_button = gr.Button("生成音频")
        # NOTE: 必须 streaming=False。streaming=True 会让 Gradio 以 HLS/ADTS(m3u8+AAC)
        # 分片形式返回音频, pyvideotrans 通过 gradio_client 调用时只能解析单个 wav 文件,
        # 收到 m3u8 后会把每条 TTS 都判为失败, 触发 "[CosyVoice(本地)] 配音全部失败 None"。
        # 详见 https://pyvideotrans.com/cosyvoice 中的"修改版 webui.py"说明。
        audio_output = gr.Audio(label="audio_output", autoplay=False, streaming=False)

        # NOTE: seed_button.click 必须在 generate_button.click 之前注册,
        # 让 generate_audio 的 fn_index 与官方 CosyVoice/webui.py 保持一致(=1)。
        # 同时显式指定 api_name="generate_audio",让 pyvideotrans 等客户端通过
        # gradio_client 用 api_name='/generate_audio' 调用时能找到端点。
        seed_button.click(_generate_seed, inputs=[], outputs=seed, api_name="generate_seed")
        generate_button.click(
            _generate_audio,
            inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text,
                    prompt_wav_upload, prompt_wav_record, instruct_text,
                    seed, stream_radio, speed],
            outputs=[audio_output],
            api_name="generate_audio",
        )
    demo.queue(max_size=4, default_concurrency_limit=2)
    return demo


def _launch_cosyvoice_webui_inproc(port: int = 8000, max_retries: int = 12, retry_interval: float = 5.0):
    """在同一进程内启动 CosyVoice WebUI 的 API 通道(供 pyvideotrans 等客户端调用)。

    复用已经加载的 cosyvoice_model,共享 GPU 显存与 TRT 引擎;不弹浏览器、不暴露
    可见 UI,只跑 Gradio 的 /run/predict 等 API endpoint。

    重试策略:
      - 端口被占用 (EADDRINUSE / OSError) 通常发生在 wsl --shutdown 之后,旧 socket
        还在 TIME_WAIT(默认 60s),systemd 自动重启会撞上。这里在进程内重试
        max_retries 次, 每次间隔 retry_interval 秒, 总等待 ~60s 足以覆盖 TIME_WAIT。
      - 重试全部失败则 sys.exit(1), 由 systemd 的 Restart=always 接管再起一轮。
      - 其他类型异常 (例如模型未加载) 不重试, 仅打印不退出, 让 7860 仍可用。
    """
    global cosyvoice_model
    if cosyvoice_model is None:
        print("[CosyVoice WebUI] cosyvoice_model 未初始化,跳过 8000 端口启动")
        return None
    import tempfile
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            cosy_demo = _build_cosyvoice_api_demo(cosyvoice_model)
            cosy_demo.launch(
                server_name='0.0.0.0',
                server_port=port,
                show_error=True,
                inbrowser=False,
                prevent_thread_lock=True,
                quiet=True,
                allowed_paths=[tempfile.gettempdir()],
            )
            print(f"[CosyVoice WebUI] API launched on http://0.0.0.0:{port} (attempt {attempt}/{max_retries})")
            return cosy_demo
        except OSError as exc:
            # 端口冲突类错误: 等一会重试
            last_exc = exc
            print(f"[CosyVoice WebUI] attempt {attempt}/{max_retries} bind {port} failed: {exc}")
            if attempt < max_retries:
                print(f"[CosyVoice WebUI] retrying in {retry_interval}s (waiting for TIME_WAIT to clear)...")
                time.sleep(retry_interval)
        except Exception as exc:  # noqa: BLE001
            # 非端口类错误: 不重试, 但也不拖垮 7860
            print(f"[CosyVoice WebUI] non-bind error on launch: {exc}")
            traceback.print_exc()
            return None
    # 所有重试都失败 -> 主动退出, 让 systemd 接管重启
    print(
        f"[CosyVoice WebUI] FATAL: failed to bind {port} after {max_retries} attempts: {last_exc}",
        file=sys.stderr,
    )
    print("[CosyVoice WebUI] exiting so systemd can Restart=always", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    print("Starting CosyVoice API (headless, 8000)...")
    # 去掉 gradio 分享链接提示 (headless 不弹 UI)
    strings_attr = getattr(gradio_module, "strings", None)
    if strings_attr is not None:
        en_strings = getattr(strings_attr, "en", None)
        if isinstance(en_strings, dict):
            en_strings["SHARE_LINK_MESSAGE"] = ""

    # 提前加载 CosyVoice 模型, 8000 API 启动后立即可用
    print("Pre-loading CosyVoice model...")
    print(load_model())

    # 启动 CosyVoice 官方 WebUI(8000), 供 pyvideotrans 等客户端调用
    cosyvoice_api_handle = _launch_cosyvoice_webui_inproc(port=8000)
    if cosyvoice_api_handle is None:
        print(
            "[CosyVoice WebUI] API failed to start; exiting for systemd Restart=always",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Server status:")
    print("  - CosyVoice WebUI API: http://0.0.0.0:8000 [OK]")
    # 主线程阻塞, 保持 server 运行; systemd 发送 SIGTERM 时退出
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
