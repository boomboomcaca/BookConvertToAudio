import gradio as gradio_module
import inspect
import os
import sys
import time
import subprocess
import threading
import json
import traceback
from datetime import datetime
from typing import Any, cast

import torch  # type: ignore[import]
import torchaudio  # type: ignore[import]

gr = cast(Any, gradio_module)

# 设置环境路径
current_dir = os.getcwd()
# 回退到标准的 CosyVoice 目录
cosyvoice_root = os.path.join(current_dir, 'CosyVoice')
sys.path.append(cosyvoice_root)
sys.path.append(os.path.join(cosyvoice_root, 'third_party', 'Matcha-TTS'))
# 如果有 ffmpeg，添加路径 (假设在 ffmpeg/bin 下，如果没有则忽略)
ffmpeg_path = os.path.join(cosyvoice_root, 'ffmpeg', 'bin')
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

# 禁用 DeepSpeed 检查
os.environ["DS_SKIP_CUDA_CHECK"] = "1"
os.environ["DS_BUILD_OPS"] = "0"

# 全局模型变量
cosyvoice_model: Any = None
# 当前模型版本标识：'v3' / 'v2' / 'v1'，用于在推理时按版本调整 prompt 文本格式
cosyvoice_model_version: str = ''

# CosyVoice 3 推理时需要在 prompt_text 前添加的系统提示
COSYVOICE3_SYSTEM_PROMPT = 'You are a helpful assistant.<|endofprompt|>'

# 是否优先加载 RL 微调过的 LLM 权重（llm.rl.pt）：质量明显优于 base llm.pt（CER 0.81% vs 1.21%）
USE_RL_LLM_IF_AVAILABLE = True

# 是否启用 TensorRT 加速 flow.decoder（DiT）。首次启动需 ~30 秒编译 .plan 引擎（一次性），
# 之后每次启动直接复用。FP16+TRT 模式可同时利用 RTX 30 系 Tensor Core + TRT 算子融合，
# 实测最快；官方 README 的 "fp16 engine performance issue" 主要指数值稳定性提示，听感上无明显劣化。
USE_TRT_IF_AVAILABLE = True

# 优先使用的模型目录（按顺序尝试）。CosyVoice3 优先
COSYVOICE_MODEL_DIR_CANDIDATES = [
    'Fun-CosyVoice3-0.5B',
    'CosyVoice2-0.5B',
    'CosyVoice-300M',
    'CosyVoice-300M-SFT',
    'CosyVoice-300M-Instruct',
]

# 停止标志
stop_flag = threading.Event()
current_inference_thread = None
# 后台任务线程
background_task_thread = None
# 任务管理锁：防止并发请求导致 background_task_thread 被覆盖
background_task_lock = threading.Lock()

# CosyVoice 模型推理锁，防止并发推理导致 CUDA/模型状态损坏
cosyvoice_inference_lock = threading.Lock()

def locked_generator(gen, lock):
    lock.acquire()
    try:
        for val in gen:
            yield val
    finally:
        lock.release()

# 资源目录
ASSETS_DIR = os.path.join(current_dir, 'assets')
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)

# 日志目录
LOG_DIR = os.path.join(current_dir, 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 全局日志文件路径（与原有变量名保持一致，方便后续引用）
log_path = os.path.join(LOG_DIR, 'web_book_converter.log')

# 输出目录（生成的文件保存到这里）
OUTPUT_DIR = os.path.join(current_dir, 'output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 前端展示的文件数量上限，避免一次性加载过多文件导致页面卡顿
MAX_FILES_IN_UI = 30
MAX_FILES_IN_STATUS_MESSAGE = 10

def _limit_files_for_ui(all_files):
    """仅保留用于前端展示的最近 N 个文件"""
    if not all_files:
        return []
    return all_files[-MAX_FILES_IN_UI:]

def _update_generated_files_state(task_state, all_files):
    """更新任务状态中的文件列表和总数"""
    task_state['generated_files'] = _limit_files_for_ui(all_files)
    task_state['total_generated_files'] = len(all_files)

def _build_completion_message(all_files):
    """构建简洁的完成信息，避免一次性输出所有文件名导致界面卡顿"""
    total = len(all_files)
    if total == 0:
        return "All done! No files were generated."
    recent = [os.path.basename(f) for f in all_files[-MAX_FILES_IN_STATUS_MESSAGE:]]
    message_lines = [f"All done! Generated {total} file(s) in output folder."]
    if recent:
        message_lines.append("最近生成：")
        message_lines.extend(recent)
    if total > MAX_FILES_IN_STATUS_MESSAGE:
        message_lines.append("其余文件请在 output 目录查看。")
    return "\n".join(message_lines)

# 任务状态文件
TASK_STATE_FILE = os.path.join(current_dir, 'task_state.json')
# 文件I/O锁，防止并发写入导致文件损坏
task_state_lock = threading.Lock()

def get_reference_audio_list():
    """扫描 assets 目录下的音频文件"""
    files = []
    valid_exts = ['.wav', '.mp3', '.flac']
    if os.path.exists(ASSETS_DIR):
        for f in os.listdir(ASSETS_DIR):
            if any(f.lower().endswith(ext) for ext in valid_exts):
                files.append(f)
    return sorted(files)

def get_prompt_text_for_audio(audio_filename):
    """尝试读取同名 txt 文件的内容"""
    if not audio_filename:
        return ""
    
    base_name = os.path.splitext(audio_filename)[0]
    txt_path = os.path.join(ASSETS_DIR, base_name + '.txt')
    
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except:
            return ""
    return ""

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

def save_task_state(state):
    """保存任务状态到JSON文件（线程安全）"""
    global task_state_lock
    try:
        with task_state_lock:
            # 使用临时文件+原子重命名，确保写入的原子性
            temp_file = TASK_STATE_FILE + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            # 原子重命名（在POSIX系统上是原子操作）
            os.replace(temp_file, TASK_STATE_FILE)
    except Exception as e:
        print(f"Failed to save task state: {e}")

def load_task_state():
    """从JSON文件加载任务状态（线程安全）"""
    global task_state_lock
    try:
        with task_state_lock:
            if os.path.exists(TASK_STATE_FILE):
                with open(TASK_STATE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
    except Exception as e:
        print(f"Failed to load task state: {e}")
    return None

def clear_task_state():
    """清除任务状态文件（线程安全）"""
    global task_state_lock
    try:
        with task_state_lock:
            if os.path.exists(TASK_STATE_FILE):
                os.remove(TASK_STATE_FILE)
    except Exception as e:
        print(f"Failed to clear task state: {e}")

def get_task_status():
    """获取当前任务状态（用于界面刷新）"""
    state = load_task_state()
    if not state:
        return "暂无运行中的任务", None
    
    status = state.get('status', 'unknown')
    current_file = state.get('current_file', '')
    total_files = state.get('total_files', 0)
    file_idx = state.get('file_idx', 0)
    progress_pct = state.get('progress', 0) * 100
    message = state.get('message', '')
    raw_generated_files = state.get('generated_files', [])
    total_generated = state.get('total_generated_files', len(raw_generated_files))
    displayed_files = _limit_files_for_ui(raw_generated_files)
    hidden_count = max(0, total_generated - len(displayed_files))
    
    if status == 'running':
        status_msg = f"🟢 任务运行中\n"
        if total_files > 0:
            status_msg += f"进度: {file_idx + 1}/{total_files} 文件 ({progress_pct:.1f}%)\n"
        if current_file:
            status_msg += f"当前文件: {current_file}\n"
        if message:
            status_msg += f"状态: {message}"
    elif status == 'completed':
        status_msg = f"✅ 任务已完成\n"
        if message:
            status_msg += f"{message}"
    elif status == 'stopped':
        status_msg = f"🛑 任务已停止\n"
        if message:
            status_msg += f"{message}"
    elif status == 'error':
        status_msg = f"❌ 任务出错\n"
        if message:
            status_msg += f"{message}"
    else:
        status_msg = f"状态: {status}\n"
        if message:
            status_msg += f"{message}"
    
    # 如果有生成的文件，返回文件列表
    files = None
    existing_files = []
    missing_count = 0
    if displayed_files:
        # 检查文件是否存在（支持相对路径和绝对路径）
        for f in displayed_files:
            # 如果是相对路径，尝试在当前目录查找
            if not os.path.isabs(f):
                full_path = os.path.join(current_dir, f)
                if os.path.exists(full_path):
                    existing_files.append(full_path)
            elif os.path.exists(f):
                existing_files.append(f)
        missing_count = len(displayed_files) - len(existing_files)
        if existing_files:
            files = existing_files
    
    visible_count = len(existing_files)
    if total_generated > 0:
        status_msg += f"\n已生成 {total_generated} 个文件"
        if visible_count > 0:
            status_msg += f"（仅显示最近 {visible_count} 个"
            if hidden_count > 0:
                status_msg += "，更多文件请查看 output 目录"
            status_msg += "）"
        elif hidden_count > 0:
            status_msg += "（最近文件已被移动或删除，请直接在 output 目录查看）"
        else:
            status_msg += "（生成的文件已不存在，可能已被移动或删除）"
    
    if missing_count > 0 and visible_count > 0:
        status_msg += f"\n有 {missing_count} 个最近文件已不存在，已自动从列表中移除。"
    
    return status_msg, files

def _execute_conversion_task(text_files, ref_audio_name, prompt_text):
    """
    在后台线程中执行转换任务（不依赖 yield，即使前端关闭也能继续运行）
    """
    global cosyvoice_model, stop_flag, current_inference_thread
    
    current_inference_thread = threading.current_thread()
    
    # 初始化任务状态
    task_state = {
        'status': 'running',
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'current_file': '',
        'file_idx': 0,
        'total_files': 0,
        'progress': 0.0,
        'message': '任务启动中...',
        'generated_files': [],
        'total_generated_files': 0
    }
    save_task_state(task_state)
    
    try:
        # 1. 检查模型状态
        if cosyvoice_model is None:
            msg = "Loading model..."
            task_state['message'] = msg
            task_state['progress'] = 0.0
            save_task_state(task_state)
            msg = load_model()
            task_state['message'] = msg
            save_task_state(task_state)
            if cosyvoice_model is None:
                task_state['status'] = 'error'
                task_state['message'] = '模型加载失败'
                save_task_state(task_state)
                return
            if stop_flag.is_set():
                msg = "转换已停止"
                task_state['status'] = 'stopped'
                task_state['message'] = msg
                save_task_state(task_state)
                print("模型加载后检测到停止标志，开始清理模型...")
                _cleanup_model_immediate()
                return

        # 2. 验证输入
        msg = "Validating inputs..."
        task_state['message'] = msg
        task_state['progress'] = 0.1
        save_task_state(task_state)
        
        if not text_files:
            task_state['status'] = 'error'
            task_state['message'] = "Error: Please upload at least one text file."
            save_task_state(task_state)
            return
        
        if not ref_audio_name:
            task_state['status'] = 'error'
            task_state['message'] = "Error: Please select a reference audio."
            save_task_state(task_state)
            return

        ref_audio_path = os.path.join(ASSETS_DIR, ref_audio_name)
        if not os.path.exists(ref_audio_path):
            task_state['status'] = 'error'
            task_state['message'] = f"Error: Audio file not found: {ref_audio_path}"
            save_task_state(task_state)
            return

        # 确保 text_files 是列表
        if not isinstance(text_files, list):
            text_files = [text_files]
        
        total_files = len(text_files)
        all_generated_files = []
        
        # 更新任务状态
        task_state['total_files'] = total_files
        _update_generated_files_state(task_state, all_generated_files)
        save_task_state(task_state)

        for file_idx, text_file in enumerate(text_files):
            # 检查停止标志
            if stop_flag.is_set():
                msg = "转换已停止"
                task_state['status'] = 'stopped'
                task_state['message'] = msg
                save_task_state(task_state)
                print("文件循环中检测到停止标志，开始清理模型...")
                _cleanup_model_immediate()
                break
                
            file_name = os.path.basename(text_file.name)
            msg = f"Processing file {file_idx + 1}/{total_files}: {file_name}..."
            task_state['current_file'] = file_name
            task_state['file_idx'] = file_idx
            task_state['message'] = msg
            task_state['progress'] = file_idx / total_files
            save_task_state(task_state)

            # 读取文本
            with open(text_file.name, 'r', encoding='utf-8') as f:
                full_text = f.read().strip()
            
            msg = f"File {file_idx + 1}/{total_files}: Text loaded ({len(full_text)} chars). Inferencing..."
            task_state['message'] = msg
            save_task_state(task_state)
            
            if stop_flag.is_set():
                msg = "转换已停止"
                task_state['status'] = 'stopped'
                task_state['message'] = msg
                save_task_state(task_state)
                print("推理开始前检测到停止标志，开始清理模型...")
                _cleanup_model_immediate()
                break
            
            # 注意：CosyVoiceFrontEnd 内部会再次调用 load_wav(prompt_wav, ...)，
            # 因此必须传入「音频文件路径」而非已加载的 Tensor，否则 torchaudio.load 会因
            # 接收到 Tensor 抛出 "Invalid file: tensor(...)" 错误。
            prompt_wav_for_inference = ref_audio_path

            # CosyVoice 3 要求 prompt_text 形如 "You are a helpful assistant.<|endofprompt|>{ref_text}"
            # 当用户未提供 system 段时自动补齐，避免推理质量下降
            effective_prompt_text = prompt_text or ''
            if cosyvoice_model_version == 'v3' and '<|endofprompt|>' not in effective_prompt_text:
                effective_prompt_text = COSYVOICE3_SYSTEM_PROMPT + effective_prompt_text

            start_time = time.time()
            
            # 3. 初始化流式累积变量
            chunk_count = 0
            estimated_chunks = max(1, len(full_text) // 10)
            
            current_part_audio = []
            current_part_samples = 0
            part_index = 0
            MAX_DURATION_SEC = 45 * 60  # 45 minutes
            sample_rate = cosyvoice_model.sample_rate
            MAX_SAMPLES = MAX_DURATION_SEC * sample_rate
            
            PAUSE_DURATION_MS = 200
            pause_samples = int(sample_rate * PAUSE_DURATION_MS / 1000)
            silence = torch.zeros(1, pause_samples)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = os.path.splitext(file_name)[0]

            # 定义内部保存函数
            def save_current_part(audio_chunks, p_idx):
                if not audio_chunks:
                    return
                
                # 拼接片段和停顿
                audio_with_pauses = []
                for idx, chunk in enumerate(audio_chunks):
                    audio_with_pauses.append(chunk)
                    if idx < len(audio_chunks) - 1:
                        audio_with_pauses.append(silence)
                
                full_part_tensor = torch.cat(audio_with_pauses, dim=1)
                
                # 生成文件名
                part_suffix = f"_part{p_idx + 1}"
                output_base = f"{base_filename}_{timestamp}{part_suffix}"
                temp_wav = os.path.join(current_dir, f"temp_{output_base}.wav")
                output_mp4 = f"{output_base}.mp4"
                mp4_path = os.path.join(OUTPUT_DIR, output_mp4)
                
                torchaudio.save(temp_wav, full_part_tensor, sample_rate)
                
                # FFmpeg 转换
                msg = f"File {file_idx + 1}/{total_files}: Converting part {p_idx + 1} to video..."
                task_state['message'] = msg
                save_task_state(task_state)
                
                # 获取音频时长，确保视频长度精确匹配，避免末尾静音
                audio_duration = None
                try:
                    probe_cmd = [
                        "ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                        temp_wav
                    ]
                    probe_result = subprocess.run(probe_cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
                    if probe_result.returncode == 0 and probe_result.stdout.strip():
                        audio_duration = float(probe_result.stdout.strip())
                except (subprocess.TimeoutExpired, ValueError, Exception) as e:
                    # 如果获取时长失败，使用 -shortest 作为备选方案
                    print(f"Warning: Could not get audio duration: {e}, using -shortest instead")
                
                # 使用更高的帧率（25 fps）以获得更精确的时长控制
                # 如果成功获取音频时长，使用 -t 参数精确控制输出时长
                cmd = [
                    "ffmpeg", "-y", "-nostdin",
                    "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25",
                    "-i", temp_wav,
                    "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-crf", "40", "-preset", "veryfast",
                    "-c:a", "aac", "-b:a", "96k",
                ]
                
                if audio_duration is not None and audio_duration > 0:
                    # 使用 -t 参数精确限制输出时长，避免末尾静音
                    cmd.extend(["-t", str(audio_duration)])
                else:
                    # 备选方案：使用 -shortest（当无法获取时长时）
                    cmd.append("-shortest")
                
                cmd.append(mp4_path)
                
                # NOTE: 必须传 stdin=DEVNULL,否则后台启动(start.sh &)时 ffmpeg 会因尝试读取
                # 父进程 stdin 触发 SIGTTIN 进入 T(stopped) 暂停状态,导致整段转换无限期挂起。
                # 同时在 cmd 中加 -nostdin 作为第二道防线。
                process = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

                if process.returncode != 0:
                     stderr_text = process.stderr.decode(errors='ignore').strip()
                     error_msg = f"FFmpeg error part {p_idx + 1}: {stderr_text}"
                     print(error_msg)
                     msg = f"File {file_idx + 1}/{total_files}: Video generation failed for part {p_idx + 1}. Details: {stderr_text}"
                     task_state['message'] = msg
                     save_task_state(task_state)
                else:
                    all_generated_files.append(mp4_path)
                    _update_generated_files_state(task_state, all_generated_files)
                    save_task_state(task_state)

            try:
                for i, output in enumerate(locked_generator(cosyvoice_model.inference_zero_shot(full_text, effective_prompt_text, prompt_wav_for_inference, stream=False), cosyvoice_inference_lock)):
                    # 检查停止标志
                    if stop_flag.is_set():
                        msg = "转换已停止，正在清理资源..."
                        task_state['message'] = msg
                        save_task_state(task_state)
                        print("推理过程中检测到停止标志，开始清理模型...")
                        _cleanup_model_immediate()
                        break
                    
                    chunk_count += 1
                    audio_chunk = output['tts_speech']
                    chunk_len = audio_chunk.shape[1]
                    duration = chunk_len / sample_rate
                    
                    msg = f"File {file_idx + 1}/{total_files}: Generated chunk {chunk_count} ({duration:.2f}s)..."
                    
                    # 进度条逻辑
                    file_progress = min(0.95, chunk_count / estimated_chunks)
                    global_progress = (file_idx + file_progress) / total_files
                    task_state['message'] = msg
                    task_state['progress'] = global_progress
                    save_task_state(task_state)
                    
                    # 累积音频
                    if current_part_audio:
                        current_part_samples += pause_samples  # 只有在片段之间才插入停顿
                    current_part_audio.append(audio_chunk)
                    current_part_samples += chunk_len
                    
                    # 检查是否达到分段阈值
                    if current_part_samples >= MAX_SAMPLES:
                        save_current_part(current_part_audio, part_index)
                        part_index += 1
                        current_part_audio = []
                        current_part_samples = 0
                        
                        # 强制清理一下内存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
            except Exception as e:
                if stop_flag.is_set():
                    msg = "转换已停止"
                    task_state['status'] = 'stopped'
                    task_state['message'] = msg
                    save_task_state(task_state)
                    print("推理异常时检测到停止标志，开始清理模型...")
                    _cleanup_model_immediate()
                    return
                raise

            if stop_flag.is_set():
                msg = "转换已停止"
                task_state['status'] = 'stopped'
                task_state['message'] = msg
                save_task_state(task_state)
                print("推理完成后检测到停止标志，开始清理模型...")
                _cleanup_model_immediate()
                return

            # 保存剩余的部分
            if current_part_audio:
                save_current_part(current_part_audio, part_index)
            elif chunk_count == 0:
                msg = f"File {file_idx + 1}/{total_files}: Error: No audio generated for {file_name}"
                task_state['message'] = msg
                save_task_state(task_state)
            
            file_time = time.time() - start_time
            msg = f"File {file_idx + 1}/{total_files}: Done ({file_time:.2f}s)"
            task_state['message'] = msg
            task_state['progress'] = (file_idx + 1) / total_files
            _update_generated_files_state(task_state, all_generated_files)
            save_task_state(task_state)

        if stop_flag.is_set():
            msg = "转换已停止"
            task_state['status'] = 'stopped'
            task_state['message'] = msg
            save_task_state(task_state)
            # 注意：模型清理在 finally 块中统一处理
        else:
            # 显示文件名（不包含完整路径）
            msg = _build_completion_message(all_generated_files)
            task_state['status'] = 'completed'
            task_state['message'] = msg
            task_state['progress'] = 1.0
            _update_generated_files_state(task_state, all_generated_files)
            save_task_state(task_state)

    except Exception as e:
        if stop_flag.is_set():
            msg = "转换已停止"
            task_state['status'] = 'stopped'
            task_state['message'] = msg
            save_task_state(task_state)
            # 注意：模型清理在 finally 块中统一处理
        else:
            import traceback
            error_trace = traceback.format_exc()
            print(error_trace)
            task_state['status'] = 'error'
            task_state['message'] = f"Error: {str(e)}"
            save_task_state(task_state)
    finally:
        # 清理资源（如果停止标志被设置，立即清理模型）
        if stop_flag.is_set():
            print("检测到停止标志，正在清理模型资源...")
            _cleanup_model_immediate()
        else:
            # 正常完成时只清理CUDA缓存，保留模型以便下次使用
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Bug 1 Fix: 只有在当前线程仍然是活动线程时才重置引用
        # 这防止了竞态条件：如果新任务已经启动并设置了 current_inference_thread，
        # 旧任务的 finally 块不应该覆盖它
        if current_inference_thread is threading.current_thread():
            current_inference_thread = None

def convert_book(text_files, ref_audio_name, prompt_text, progress=None):
    """
    启动转换任务（在后台线程中执行，即使前端关闭也能继续运行）
    这个函数只负责启动任务并定期报告状态
    """
    global background_task_thread, stop_flag, background_task_lock
    
    # Bug 1 Fix: 使用锁同步访问 background_task_thread，防止并发请求导致线程引用被覆盖
    # 这确保即使 default_concurrency_limit=2 允许并发请求，也只有一个任务能启动
    with background_task_lock:
        # 如果已有任务在运行，先停止它
        if background_task_thread and background_task_thread.is_alive():
            stop_flag.set()
            background_task_thread.join(timeout=2)
            # Bug 2 Fix: 如果线程仍在运行，保持停止标志设置，不重置
            # 只有在线程确实已结束时才清除标志
            if background_task_thread.is_alive():
                # 线程仍在运行，保持停止标志设置
                # 新任务不应该启动，因为旧任务还在运行
                yield "Error: Previous task is still running. Please wait for it to stop or restart the application.", None
                return
            # 线程已结束，现在可以安全地清除标志
            stop_flag.clear()
        else:
            # 没有运行中的任务，确保标志已清除
            stop_flag.clear()
    
    # 验证输入
    if not text_files:
        yield "Error: Please upload at least one text file.", None
        return
    
    if not ref_audio_name:
        yield "Error: Please select a reference audio.", None
        return

    ref_audio_path = os.path.join(ASSETS_DIR, ref_audio_name)
    if not os.path.exists(ref_audio_path):
        yield f"Error: Audio file not found: {ref_audio_path}", None
        return
    
    # 保存文件路径（因为 text_file.name 可能在后台线程中失效）
    if not isinstance(text_files, list):
        text_files = [text_files]
    
    # 保存文件路径到临时文件，供后台线程使用
    text_file_paths = []
    for text_file in text_files:
        # 如果是文件对象，保存路径
        if hasattr(text_file, 'name'):
            text_file_paths.append(text_file.name)
        else:
            text_file_paths.append(str(text_file))
    
    # 启动后台任务线程
    def run_task():
        # 重新打开文件（因为原始文件对象可能已关闭）
        file_objects = []
        for path in text_file_paths:
            if os.path.exists(path):
                # 创建一个类似文件对象的包装
                class FileWrapper:
                    def __init__(self, path):
                        self.name = path
                file_objects.append(FileWrapper(path))
        
        if file_objects:
            _execute_conversion_task(file_objects, ref_audio_name, prompt_text)
        else:
            # Bug 1 Fix: 如果没有找到任何文件，更新任务状态为错误
            task_state = {
                'status': 'error',
                'message': 'Error: No valid text files found. Files may have been deleted or moved.',
                'progress': 0.0,
                'generated_files': [],
                'total_generated_files': 0
            }
            save_task_state(task_state)
    
    # Bug 1 Fix: 在线程创建、赋值和启动时持有锁，防止并发请求覆盖 background_task_thread
    # 这确保即使两个请求同时到达，也只有一个能成功创建和启动任务线程
    with background_task_lock:
        # 再次检查（在锁内），防止在验证输入期间另一个请求已经启动了任务
        if background_task_thread and background_task_thread.is_alive():
            yield "Error: Another task was started while validating inputs. Please wait for it to complete.", None
            return
        
        # 创建并赋值线程（在锁保护下）
        background_task_thread = threading.Thread(target=run_task, daemon=False)
        # Bug 1 Fix: start() 必须在锁内调用，防止在释放锁和启动线程之间
        # 另一个请求覆盖 background_task_thread，导致启动错误的线程
        background_task_thread.start()
    
    # 初始化任务状态
    task_state = {
        'status': 'running',
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'current_file': '',
        'file_idx': 0,
        'total_files': 0,
        'progress': 0.0,
        'message': '任务启动中...',
        'generated_files': [],
        'total_generated_files': 0
    }
    save_task_state(task_state)
    
    yield "任务已启动，正在后台运行...", None
    
    # 定期报告任务状态（即使前端关闭，任务也会在后台继续运行）
    # 这个循环作为 fallback，确保在 auto-refresh 不可用时（如旧版 Gradio）仍能提供更新
    # 即使 auto-refresh 可用，这个循环也能提供更及时的更新
    last_status = None
    last_files = None
    while True:
        try:
            # 检查任务是否还在运行
            if background_task_thread and not background_task_thread.is_alive():
                # 任务已完成，获取最终状态
                status_msg, files = get_task_status()
                if status_msg != last_status or files != last_files:
                    yield status_msg, files
                break
            
            # 获取当前任务状态
            status_msg, files = get_task_status()
            if status_msg != last_status or files != last_files:
                yield status_msg, files
                last_status = status_msg
                last_files = files
            
            # 检查任务是否完成或出错
            task_state = load_task_state()
            if task_state and task_state.get('status') in ['completed', 'error', 'stopped']:
                # 任务已完成，等待线程结束
                if background_task_thread:
                    background_task_thread.join(timeout=1)
                # 获取最终状态
                status_msg, files = get_task_status()
                if status_msg != last_status or files != last_files:
                    yield status_msg, files
                break
            
            # 等待一段时间再检查（避免过于频繁）
            time.sleep(1)
            
        except GeneratorExit:
            # 前端关闭，但任务继续在后台运行
            # 不重新抛出异常，让生成器正常结束
            break
        except Exception as e:
            # 其他异常，记录但不中断任务
            print(f"Error in status reporting: {e}")
            time.sleep(1)

def refresh_audio_list():
    return gr.Dropdown(choices=get_reference_audio_list())

def _cleanup_model_immediate():
    """立即清理模型资源（同步执行，确保GPU资源释放）"""
    global cosyvoice_model
    import gc
    
    try:
        # 安全地检查模型是否存在
        model_ref = None
        try:
            model_ref = cosyvoice_model
        except Exception:
            pass
        
        if model_ref is not None:
            try:
                # 尝试将模型移到 CPU（如果支持）
                if hasattr(model_ref, 'to'):
                    try:
                        model_ref.to('cpu')
                        print("模型已移到CPU")
                    except Exception as e:
                        print(f"移动模型到CPU时出现警告: {e}")
            except Exception as e:
                print(f"检查模型移动方法时出现警告: {e}")
            
            # 尝试清理模型内部资源
            try:
                if hasattr(model_ref, 'cpu'):
                    model_ref.cpu()
            except Exception:
                pass
            
            # 删除模型引用
            try:
                del model_ref
            except Exception:
                pass
            
            # 清除全局引用
            try:
                cosyvoice_model = None
            except Exception:
                pass
            print("模型引用已清除")
        
        # 清理 CUDA 缓存
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 确保所有CUDA操作完成
                print("CUDA缓存已清理")
        except Exception as e:
            print(f"清理CUDA缓存时出现警告: {e}")
        
        # 垃圾回收
        try:
            gc.collect()
            print("垃圾回收已完成")
        except Exception as e:
            print(f"垃圾回收时出现警告: {e}")
        
        # 再次清理 CUDA（确保彻底释放）
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception as e:
        print(f"清理模型时出现错误: {e}")
        import traceback
        traceback.print_exc()
        # 即使出错也尝试清除全局引用
        try:
            cosyvoice_model = None
        except Exception:
            pass

def _cleanup_model_background():
    """在后台线程中清理模型资源（避免阻塞主线程）"""
    import time
    # 先等待一小段时间，确保主函数已经返回
    time.sleep(0.2)
    # 调用立即清理函数
    _cleanup_model_immediate()

def stop_conversion():
    """停止转换并强制清理资源，包括卸载模型 - 快速返回版本"""
    global stop_flag
    
    # 用 try-except 包裹整个函数，捕获所有未处理的异常
    try:
        # 第一步：立即设置停止标志（最快操作）
        try:
            if stop_flag is not None:
                stop_flag.set()
        except Exception as e:
            print(f"ERROR in stop_conversion (stop_flag.set): {e}")
            # Don't re-raise - continue execution to return message
        
        # 第二步：立即返回消息（不等待任何其他操作）
        result_msg = "🛑 转换已停止，正在清理资源..."
        
        # 第三步：所有其他操作都在后台异步执行
        def async_operations():
            """异步执行所有可能阻塞的操作"""
            global cosyvoice_model
            try:
                # 更新任务状态
                try:
                    task_state = load_task_state()
                    if task_state is not None and isinstance(task_state, dict):
                        task_state['status'] = 'stopped'
                        task_state['message'] = result_msg
                        save_task_state(task_state)
                except Exception as e:
                    print(f"ERROR in async_operations (task_state): {e}")
                    pass
                
                # 执行模型清理（立即清理，不等待）
                # 注意：转换任务也会在检测到 stop_flag 时清理模型，这里是双重保险
                print("stop_conversion: 开始后台清理模型...")
                _cleanup_model_immediate()
            except Exception as e:
                print(f"后台操作时出现警告: {e}")
        
        # 启动后台线程（不等待）
        try:
            thread = threading.Thread(target=async_operations, daemon=True)
            thread.start()
        except Exception as e:
            print(f"ERROR in stop_conversion (thread start): {e}")
            pass
        
        # 立即返回空列表（不返回消息，因为 outputs=[]，消息通过 task_state 更新显示）
        return []
    
    except Exception as e:
        # 捕获所有未处理的异常
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"id":"log_stop_013","timestamp":int(time.time()*1000),"location":"web_book_converter.py:805","message":"stop_conversion unhandled exception","data":{"error":str(e),"traceback":traceback.format_exc()},"sessionId":"debug-session","runId":"run2","hypothesisId":"A"})+'\n')
        except Exception as log_err:
            print(f"DEBUG LOG ERROR (unhandled exception): {log_err}")
        # #endregion
        print(f"CRITICAL ERROR in stop_conversion: {e}")
        traceback.print_exc()
        # 返回空列表（与 outputs=[] 配置一致，避免 Gradio 连接错误）
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"id":"log_stop_013b","timestamp":int(time.time()*1000),"location":"web_book_converter.py:868","message":"returning empty list from exception handler","data":{"error":str(e)},"sessionId":"debug-session","runId":"run5","hypothesisId":"G,B"})+'\n')
        except Exception as log_err:
            print(f"DEBUG LOG ERROR (exception return): {log_err}")
        # #endregion
        return []  # 假设G: 返回空列表与 outputs=[] 一致

# Gradio 界面构建
custom_css = """
    /* 主容器自适应 */
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        height: 100vh !important;
        max-height: 100vh !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
    }
    
    .main-container {
        padding: 0.3rem !important;
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        min-height: 0 !important;
        max-height: 100% !important;
    }
    
    /* 标题自适应 */
    .markdown {
        margin: 0.2rem 0 !important;
        font-size: clamp(0.9rem, 2vw, 1.1rem) !important;
        flex-shrink: 0 !important;
    }
    
    /* 表单元素自适应 */
    .form {
        margin-bottom: 0.3rem !important;
        flex-shrink: 0 !important;
    }
    
    .panel {
        margin-bottom: 0.3rem !important;
        flex-shrink: 0 !important;
    }
    
    /* 文件上传区域自适应 */
    .file-upload-area {
        min-height: 60px !important;
        height: auto !important;
        max-height: 150px !important;
        overflow-y: auto !important;
    }
    
    /* 文本框自适应 */
    .textbox {
        min-height: auto !important;
        height: auto !important;
        resize: vertical !important;
    }
    
    /* 按钮自适应 */
    .button {
        height: auto !important;
        min-height: 32px !important;
        padding: 0.3rem 0.8rem !important;
        white-space: nowrap !important;
    }
    
    .accordion {
        margin-bottom: 0.3rem !important;
        flex-shrink: 0 !important;
    }
    
    /* 行布局自适应 */
    .gradio-row {
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
        align-items: flex-start !important;
    }
    
    /* 列布局自适应 */
    .gradio-column {
        display: flex !important;
        flex-direction: column !important;
        min-height: fit-content !important;
        height: auto !important;
        flex: 1 1 auto !important;
    }
    
    /* 输出文本框自适应 */
    .output-textbox {
        flex: 1 1 auto !important;
        min-height: 100px !important;
        max-height: 40vh !important;
        overflow-y: auto !important;
        resize: vertical !important;
    }
    
    /* 文件下载区域自适应 */
    .file-download {
        flex-shrink: 0 !important;
        max-height: 25vh !important;
        overflow-y: auto !important;
        min-height: 60px !important;
    }
    
    /* 响应式布局：小屏幕时上下堆叠 */
    @media (max-width: 768px) {
        .gradio-row {
            flex-direction: column !important;
        }
        
        .gradio-column {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
        }
        
        .output-textbox {
            max-height: 30vh !important;
        }
        
        .file-download {
            max-height: 20vh !important;
        }
    }
    
    /* 超小屏幕优化 */
    @media (max-width: 480px) {
        .main-container {
            padding: 0.2rem !important;
        }
        
        .button {
            min-height: 36px !important;
            font-size: 0.9rem !important;
        }
        
        .output-textbox {
            max-height: 25vh !important;
        }
    }
    
    /* 高屏幕优化 */
    @media (min-height: 900px) {
        .output-textbox {
            max-height: 50vh !important;
        }
    }
"""

with gr.Blocks(title="CosyVoice3 Book Converter", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("### 📚 CosyVoice 3 有声书转换器")
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=280):
            text_input = gr.File(
                label="上传书籍 (.txt)", 
                file_types=[".txt"], 
                file_count="multiple"
            )
            
            with gr.Accordion("参考音频设置", open=False):
                with gr.Row():
                    ref_audio_dropdown = gr.Dropdown(
                        label="参考音频", 
                        choices=get_reference_audio_list(),
                        value=get_reference_audio_list()[0] if get_reference_audio_list() else None,
                        interactive=True,
                        scale=4,
                        container=False
                    )
                    refresh_btn = gr.Button("🔄", size="sm", scale=1, min_width=40)
                
                prompt_text_input = gr.Textbox(
                    label="Prompt Text", 
                    lines=1,
                    placeholder="选择音频后自动填充...",
                    max_lines=1,
                    container=False
                )
            
            with gr.Row():
                convert_btn = gr.Button("开始转换", variant="primary", scale=1, size="sm")
                stop_btn = gr.Button("停止转换", variant="stop", scale=1, size="sm")
        
        with gr.Column(scale=1, min_width=280):
            with gr.Row():
                log_output = gr.Textbox(
                    label="运行日志", 
                    lines=6, 
                    interactive=False,
                    container=False,
                    elem_classes=["output-textbox"],
                    scale=4
                )
                refresh_log_btn = gr.Button("🔄 刷新日志", size="sm", scale=1, min_width=80)
            files_output = gr.File(
                label="生成文件下载", 
                file_count="multiple", 
                interactive=False,
                elem_classes=["file-download"]
            )

    # 事件绑定
    refresh_btn.click(fn=refresh_audio_list, inputs=[], outputs=ref_audio_dropdown)
    
    # 选择音频时自动更新 prompt text
    ref_audio_dropdown.change(
        fn=get_prompt_text_for_audio,
        inputs=[ref_audio_dropdown],
        outputs=[prompt_text_input]
    )

    submit_event = convert_btn.click(
        fn=convert_book,
        inputs=[text_input, ref_audio_dropdown, prompt_text_input],
        outputs=[log_output, files_output]
    )
    
    # 停止按钮：调用停止函数并取消事件
    # 使用 show_progress=False 确保立即响应
    # 不输出到 log_output 以避免与 convert_book 的并发更新冲突
    # 停止消息会通过任务状态更新，由 get_task_status 自动显示
    # 
    # 修复：创建包装函数以确保 Gradio 正确处理返回值
    def stop_conversion_wrapper():
        """包装函数：确保 Gradio 正确处理停止操作"""
        try:
            stop_conversion()
            return []  # outputs=[] 时应该返回空列表
        except Exception as e:
            print(f"ERROR in stop_conversion_wrapper: {e}")
            import traceback
            traceback.print_exc()
            return []  # 即使异常也返回空列表（与 outputs=[] 一致）
    
    # 修复：使用 inputs=[] 和 outputs=[] 而不是 None，确保 Gradio 正确处理返回值
    stop_btn.click(
        fn=stop_conversion_wrapper,
        inputs=[],
        outputs=[],
        show_progress=False
    )
    
    # 刷新日志按钮 - 同时刷新任务状态
    refresh_log_btn.click(
        fn=get_task_status,
        inputs=None,
        outputs=[log_output, files_output]
    )
    
    # 初始化时尝试加载第一个音频的文本，并恢复任务状态
    def init_ui(ref_audio):
        """初始化界面：加载音频文本和恢复任务状态"""
        prompt_text = get_prompt_text_for_audio(ref_audio)
        status_msg, files = get_task_status()
        return prompt_text, status_msg, files
    
    demo.load(
        fn=init_ui,
        inputs=[ref_audio_dropdown], 
        outputs=[prompt_text_input, log_output, files_output]
    )
    
    # 注意：自动刷新功能需要 Gradio 4.0+，如果版本不支持会报错
    # 已提供手动刷新按钮，用户可以通过点击"🔄 刷新日志"按钮来查看最新状态
    # 如果需要自动刷新，请升级 Gradio: pip install --upgrade gradio>=4.0.0
    try:
        demo.load(
            fn=get_task_status,
            inputs=None,
            outputs=[log_output, files_output],
            every=5  # 每5秒自动刷新（需要 Gradio 4.0+）
        )
    except TypeError:
        # Gradio 版本不支持 every 参数，跳过自动刷新
        # 用户可以使用手动刷新按钮
        pass

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
    print("Starting Web UI...")
    # 使用 0.0.0.0 让服务在所有网络接口上监听
    strings_attr = getattr(gradio_module, "strings", None)
    if strings_attr is not None:
        en_strings = getattr(strings_attr, "en", None)
        if isinstance(en_strings, dict):
            en_strings["SHARE_LINK_MESSAGE"] = ""

    # 提前加载 CosyVoice 模型,这样 8000 端口的 API 启动后立即可用(否则要等首次调用)
    print("Pre-loading CosyVoice model (shared by 7860 GUI and 8000 API)...")
    print(load_model())

    # 启动 CosyVoice 官方 WebUI(8000),共用同一份模型,供 pyvideotrans 等客户端调用
    # 注意: 失败重试 12 次仍失败的话 _launch_cosyvoice_webui_inproc 会 sys.exit(1)
    cosyvoice_api_handle = _launch_cosyvoice_webui_inproc(port=8000)

    try:
        # 使用 queue() 启用任务队列，确保任务在后台继续运行
        # max_size=3 允许最多 3 个并发请求，确保停止按钮和刷新按钮可以响应
        demo.queue(max_size=3, default_concurrency_limit=2).launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            quiet=False,
            inbrowser=False,
            prevent_thread_lock=True,
        )
    except ValueError as e:
        if "shareable link" in str(e):
            print("Fallback: Using share=True due to network restrictions")
            demo.queue(max_size=3, default_concurrency_limit=2).launch(
                server_name="0.0.0.0",
                server_port=7860,
                show_error=True,
                share=True,
                inbrowser=False,
                prevent_thread_lock=True,
            )
        else:
            raise

    # 主线程阻塞,保持两个 Gradio server 一直运行;systemd 发送 SIGTERM 时直接退出
    print("Servers status:")
    print("  - Book Converter GUI: http://0.0.0.0:7860 [OK]")
    if cosyvoice_api_handle is not None:
        print("  - CosyVoice WebUI API: http://0.0.0.0:8000 [OK]")
    else:
        print("  - CosyVoice WebUI API: NOT RUNNING (model not loaded; pyvideotrans cannot connect)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")