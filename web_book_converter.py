import gradio as gr
import os
import sys
import torch
import torchaudio
import time
import subprocess
from datetime import datetime

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
cosyvoice_model = None

# 资源目录
ASSETS_DIR = os.path.join(current_dir, 'assets')
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)

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

def load_model():
    global cosyvoice_model
    if cosyvoice_model is None:
        try:
            # 尝试适配不同的代码版本
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice2 as CosyVoiceCls
                print("Detected CosyVoice2 class.")
            except ImportError:
                from cosyvoice.cli.cosyvoice import CosyVoice as CosyVoiceCls
                print("Detected CosyVoice class.")

            # 指向刚下载的 CosyVoice2-0.5B
            model_dir = os.path.join(cosyvoice_root, 'pretrained_models', 'CosyVoice2-0.5B')
            print(f"Loading model from {model_dir}...")
            
            try:
                # 尝试加载
                cosyvoice_model = CosyVoiceCls(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=True)
                return "Model loaded successfully (FP16)."
            except Exception as e:
                print(f"FP16 load failed: {e}, trying FP32...")
                cosyvoice_model = CosyVoiceCls(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
                return "Model loaded successfully (FP32)."
        except Exception as e:
            return f"Error loading model: {str(e)}"
    return "Model already loaded."

def convert_book(text_files, ref_audio_name, prompt_text, progress=gr.Progress(track_tqdm=True)):
    global cosyvoice_model
    
    # 1. 检查模型状态
    if cosyvoice_model is None:
        progress(0, desc="Loading model...")
        yield "Loading model...", None
        msg = load_model()
        yield msg, None
        if cosyvoice_model is None:
            return

    # 2. 验证输入
    progress(0.1, desc="Validating inputs...")
    yield "Validating inputs...", None
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

    # 确保 text_files 是列表
    if not isinstance(text_files, list):
        text_files = [text_files]

    total_files = len(text_files)
    all_generated_files = []
    last_mp4_path = None

    try:
        for file_idx, text_file in enumerate(text_files):
            file_name = os.path.basename(text_file.name)
            progress((file_idx) / total_files, desc=f"Processing file {file_idx + 1}/{total_files}: {file_name}")
            yield f"Processing file {file_idx + 1}/{total_files}: {file_name}...", None

            # 读取文本
            with open(text_file.name, 'r', encoding='utf-8') as f:
                full_text = f.read().strip()
            
            yield f"File {file_idx + 1}/{total_files}: Text loaded ({len(full_text)} chars). Inferencing...", None
            
            from cosyvoice.utils.file_utils import load_wav
            prompt_speech_16k = load_wav(ref_audio_path, 16000)

            all_audio = []
            start_time = time.time()
            
            # 3. 开始推理
            chunk_count = 0
            # 粗略估算总 chunks 数：假设每 10 个字符生成一个 chunk（根据经验值调整）
            estimated_chunks = max(1, len(full_text) // 10)
            
            for i, output in enumerate(cosyvoice_model.inference_zero_shot(full_text, prompt_text, prompt_speech_16k, stream=False)):
                chunk_count += 1
                duration = output['tts_speech'].shape[1] / 24000
                msg = f"File {file_idx + 1}/{total_files}: Generated chunk {chunk_count} ({duration:.2f}s)..."
                yield msg, None
                
                # 计算当前文件内的进度 (0.0 - 0.9)，预留 0.1 给视频转换
                file_progress = min(0.9, chunk_count / estimated_chunks)
                global_progress = (file_idx + file_progress) / total_files
                progress(global_progress, desc=msg)
                
                all_audio.append(output['tts_speech'])

            if not all_audio:
                yield f"File {file_idx + 1}/{total_files}: Error: No audio generated for {file_name}", None
                continue

            # 4. 处理结果（按最大时长拆分）
            MAX_DURATION_SEC = 45 * 60  # 45 minutes
            
            full_audio_tensor = torch.cat(all_audio, dim=1)
            total_samples = full_audio_tensor.shape[1]
            sample_rate = cosyvoice_model.sample_rate
            max_samples = MAX_DURATION_SEC * sample_rate
            
            num_parts = (total_samples + max_samples - 1) // max_samples
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 使用文件名作为输出前缀
            base_filename = os.path.splitext(file_name)[0]
            
            for i in range(num_parts):
                progress((file_idx + (i / num_parts)) / total_files, desc=f"Converting part {i+1}/{num_parts} to video...")
                
                start = i * max_samples
                end = min((i + 1) * max_samples, total_samples)
                part_tensor = full_audio_tensor[:, start:end]
                
                part_suffix = f"_part{i+1}" if num_parts > 1 else ""
                output_base = f"{base_filename}_{timestamp}{part_suffix}"
                
                temp_wav = os.path.join(current_dir, f"temp_{output_base}.wav")
                output_mp4 = f"{output_base}.mp4"
                mp4_path = os.path.join(current_dir, output_mp4)
                
                torchaudio.save(temp_wav, part_tensor, sample_rate)
                
                # 5. 生成视频 (FFmpeg)
                yield f"File {file_idx + 1}/{total_files}: Converting part {i+1}/{num_parts} to video...", None
                
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=320x240:r=1",
                    "-i", temp_wav,
                    "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-crf", "40", "-preset", "veryfast",
                    "-c:a", "aac", "-b:a", "64k",
                    "-shortest",
                    mp4_path
                ]
                
                process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

                if process.returncode != 0:
                     print(f"FFmpeg error part {i+1}: {process.stderr.decode()}")
                     yield f"File {file_idx + 1}/{total_files}: Video generation failed for part {i+1}.", None
                else:
                    all_generated_files.append(output_mp4)
            
            file_time = time.time() - start_time
            yield f"File {file_idx + 1}/{total_files}: Done ({file_time:.2f}s)", all_generated_files

        progress(1.0, desc="All done!")
        msg = f"All done! Generated {len(all_generated_files)} file(s):\n" + "\n".join(all_generated_files)
        yield msg, all_generated_files

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"Error: {str(e)}", None

def refresh_audio_list():
    return gr.Dropdown(choices=get_reference_audio_list())

# Gradio 界面构建
with gr.Blocks(title="CosyVoice Book Converter") as demo:
    gr.Markdown("# 📚 CosyVoice 有声书转换器")
    gr.Markdown("上传 txt 文本，选择预设的参考音频，一键生成有声书视频。")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.File(label="上传书籍 (.txt)", file_types=[".txt"], file_count="multiple")
            
            with gr.Group():
                gr.Markdown("### 参考音频设置")
                with gr.Row():
                    ref_audio_dropdown = gr.Dropdown(
                        label="选择参考音频 (来自 assets 文件夹)", 
                        choices=get_reference_audio_list(),
                        value=get_reference_audio_list()[0] if get_reference_audio_list() else None,
                        interactive=True
                    )
                    refresh_btn = gr.Button("🔄", size="sm", scale=0)
                
                prompt_text_input = gr.Textbox(
                    label="参考音频对应的文本 (Prompt Text)", 
                    lines=2,
                    placeholder="选择音频后自动填充..."
                )
            
            with gr.Row():
                convert_btn = gr.Button("开始转换", variant="primary")
                stop_btn = gr.Button("停止转换", variant="stop")
        
        with gr.Column():
            log_output = gr.Textbox(label="运行日志", lines=10, interactive=False)
            # video_output removed
            files_output = gr.File(label="所有生成文件下载", file_count="multiple", interactive=False)

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
    
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[submit_event])
    
    # 初始化时尝试加载第一个音频的文本
    demo.load(
        fn=get_prompt_text_for_audio,
        inputs=[ref_audio_dropdown], 
        outputs=[prompt_text_input]
    )

if __name__ == "__main__":
    print("Starting Web UI...")
    # 使用 0.0.0.0 让服务在所有网络接口上监听
    import gradio
    # 尝试禁用 localhost 检查
    gradio.strings.en["SHARE_LINK_MESSAGE"] = ""
    try:
        demo.queue().launch(
            server_name="0.0.0.0", 
            server_port=7860, 
            show_error=True,
            quiet=False,
            _frontend=False  # 禁用前端检查
        )
    except ValueError as e:
        if "shareable link" in str(e):
            print("Fallback: Using share=True due to network restrictions")
            demo.queue().launch(server_name="0.0.0.0", server_port=7860, show_error=True, share=True)
