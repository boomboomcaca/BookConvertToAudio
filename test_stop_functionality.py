#!/usr/bin/env python3
"""
Playwright 自动化测试脚本 - 测试停止转换功能

使用方法:
1. 确保应用正在运行 (http://127.0.0.1:7860)
2. 安装依赖: pip install playwright pytest-playwright
3. 安装浏览器: playwright install chromium
4. 运行测试: python test_stop_functionality.py
"""

import asyncio
import json
import os
import time
from pathlib import Path
from playwright.async_api import async_playwright, Page, expect

# 测试配置
APP_URL = "http://127.0.0.1:7860"
LOG_PATH = "/mnt/d/Soft/BookConvertToAudio/.cursor/debug.log"
TEST_TIMEOUT = 30000  # 30秒超时
WAIT_FOR_START = 5  # 等待转换开始的时间（秒）


async def wait_for_page_ready(page: Page, timeout: int = 10000):
    """等待页面完全加载"""
    try:
        # 等待 Gradio 界面加载
        await page.wait_for_selector('text=开始转换', timeout=timeout)
        await page.wait_for_selector('text=停止转换', timeout=timeout)
        print("✓ 页面加载完成")
        return True
    except Exception as e:
        print(f"✗ 页面加载失败: {e}")
        return False


async def check_for_errors(page: Page) -> list:
    """检查页面是否有错误消息"""
    errors = []
    
    # 检查控制台错误
    console_errors = []
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    
    # 检查页面中的错误文本
    error_selectors = [
        'text=Connection errored out',
        'text=Error',
        'text=错误',
        '[class*="error"]',
    ]
    
    for selector in error_selectors:
        try:
            elements = await page.query_selector_all(selector)
            if elements:
                for elem in elements:
                    text = await elem.text_content()
                    if text and text.strip():
                        errors.append(f"页面错误: {text.strip()}")
        except:
            pass
    
    return errors


async def read_debug_log() -> list:
    """读取调试日志"""
    logs = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            logs.append(json.loads(line))
                        except:
                            pass
        except Exception as e:
            print(f"读取日志文件失败: {e}")
    return logs


async def test_stop_functionality():
    """测试停止转换功能"""
    print("\n" + "="*60)
    print("开始 Playwright 自动化测试 - 停止转换功能")
    print("="*60 + "\n")
    
    # 清空之前的日志
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
        print("✓ 已清空之前的调试日志")
    
    async with async_playwright() as p:
        # 启动浏览器，添加字体和编码支持以解决中文乱码问题
        browser = await p.chromium.launch(
            headless=False,  # headless=False 可以看到浏览器操作
            args=[
                '--lang=zh-CN',  # 设置浏览器语言为中文
                '--font-render-hinting=none',  # 改善字体渲染
            ]
        )
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 720},
            locale='zh-CN',  # 设置区域为中文
            timezone_id='Asia/Shanghai',  # 设置时区
            # 设置 HTTP 头，确保服务器返回中文内容
            extra_http_headers={
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Charset': 'UTF-8'
            }
        )
        page = await context.new_page()
        
        # 设置页面编码和语言偏好
        await page.set_extra_http_headers({
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Charset': 'UTF-8'
        })
        
        test_file_path = None  # 初始化变量，避免 finally 块中的 UnboundLocalError
        try:
            # 步骤1: 访问应用
            print("\n[步骤 1] 访问应用...")
            await page.goto(APP_URL, wait_until='networkidle', timeout=30000)
            await asyncio.sleep(2)  # 等待 Gradio 初始化
            
            if not await wait_for_page_ready(page):
                print("✗ 测试失败: 页面未正确加载")
                return False
            
            print("✓ 应用访问成功")
            
            # 步骤2: 检查按钮是否存在
            print("\n[步骤 2] 检查按钮...")
            start_btn = page.get_by_role('button', name='开始转换')
            stop_btn = page.get_by_role('button', name='停止转换')
            
            await expect(start_btn).to_be_visible(timeout=5000)
            await expect(stop_btn).to_be_visible(timeout=5000)
            print("✓ 按钮检查通过")
            
            # 步骤3: 准备测试文件（创建一个简单的测试文本文件）
            print("\n[步骤 3] 准备测试文件...")
            test_file_path = Path("test_book.txt")
            test_content = "这是一个测试文本。\n" * 10  # 创建足够长的文本以触发转换
            test_file_path.write_text(test_content, encoding='utf-8')
            print(f"✓ 测试文件已创建: {test_file_path}")
            
            # 步骤4: 上传文件
            print("\n[步骤 4] 上传测试文件...")
            file_input = page.locator('input[type="file"]').first
            await file_input.set_input_files(str(test_file_path.absolute()))
            await asyncio.sleep(1)  # 等待文件上传完成
            print("✓ 文件上传成功")
            
            # 步骤5: 开始转换
            print("\n[步骤 5] 点击开始转换...")
            await start_btn.click()
            print("✓ 已点击开始转换按钮")
            
            # 步骤6: 等待转换开始（观察日志输出）
            print(f"\n[步骤 6] 等待转换开始（{WAIT_FOR_START}秒）...")
            await asyncio.sleep(WAIT_FOR_START)
            
            # 检查是否有日志输出（说明转换已开始）
            log_output = page.locator('textarea').first  # 运行日志文本框
            log_text = await log_output.input_value()
            
            if not log_text or len(log_text.strip()) < 10:
                print("⚠ 警告: 转换可能未开始，但继续测试停止功能...")
            else:
                print(f"✓ 转换已开始，日志长度: {len(log_text)} 字符")
                print(f"  日志预览: {log_text[:100]}...")
            
            # 步骤7: 点击停止按钮
            print("\n[步骤 7] 点击停止转换按钮...")
            
            # 记录点击前的时间
            before_click_time = time.time()
            
            # 点击停止按钮
            await stop_btn.click()
            print("✓ 已点击停止转换按钮")
            
            # 步骤8: 等待响应并检查错误
            print("\n[步骤 8] 等待响应并检查错误...")
            await asyncio.sleep(3)  # 等待响应
            
            # 检查是否有错误消息
            errors = await check_for_errors(page)
            
            if errors:
                print("✗ 发现错误:")
                for error in errors:
                    print(f"  - {error}")
                return False
            else:
                print("✓ 未发现错误消息")
            
            # 步骤9: 检查日志更新
            print("\n[步骤 9] 检查日志更新...")
            await asyncio.sleep(2)  # 等待日志更新
            updated_log_text = await log_output.input_value()
            
            if "停止" in updated_log_text or "stopped" in updated_log_text.lower():
                print("✓ 日志中显示停止消息")
            else:
                print("⚠ 警告: 日志中未明确显示停止消息")
                print(f"  当前日志: {updated_log_text[-200:]}")
            
            # 步骤10: 读取调试日志
            print("\n[步骤 10] 分析调试日志...")
            await asyncio.sleep(1)  # 确保所有日志都已写入
            debug_logs = await read_debug_log()
            
            if debug_logs:
                print(f"✓ 读取到 {len(debug_logs)} 条调试日志")
                
                # 查找关键日志
                stop_logs = [log for log in debug_logs if 'stop' in log.get('id', '').lower()]
                if stop_logs:
                    print(f"✓ 找到 {len(stop_logs)} 条停止相关日志")
                    for log in stop_logs[:3]:  # 显示前3条
                        print(f"  - {log.get('id')}: {log.get('message')}")
                else:
                    print("⚠ 未找到停止相关日志")
            else:
                print("⚠ 未读取到调试日志（可能日志系统未启用）")
            
            # 步骤11: 验证停止按钮状态
            print("\n[步骤 11] 验证按钮状态...")
            stop_btn_visible = await stop_btn.is_visible()
            start_btn_visible = await start_btn.is_visible()
            
            print(f"✓ 停止按钮可见: {stop_btn_visible}")
            print(f"✓ 开始按钮可见: {start_btn_visible}")
            
            # 测试完成
            print("\n" + "="*60)
            print("✓ 测试完成！")
            print("="*60 + "\n")
            
            # 保持浏览器打开一段时间以便观察
            print("保持浏览器打开 5 秒以便观察...")
            await asyncio.sleep(5)
            
            return True
            
        except Exception as e:
            print(f"\n✗ 测试过程中发生异常: {e}")
            import traceback
            traceback.print_exc()
            
            # 截图保存
            screenshot_path = "test_error_screenshot.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            print(f"✓ 错误截图已保存: {screenshot_path}")
            
            return False
            
        finally:
            # 清理测试文件
            if test_file_path and test_file_path.exists():
                test_file_path.unlink()
                print("✓ 已清理测试文件")
            
            # 关闭浏览器
            await browser.close()
            print("✓ 浏览器已关闭")


async def main():
    """主函数"""
    try:
        success = await test_stop_functionality()
        if success:
            print("\n✅ 所有测试通过！")
            exit(0)
        else:
            print("\n❌ 测试失败，请检查上述错误信息")
            exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠ 测试被用户中断")
        exit(130)
    except Exception as e:
        print(f"\n✗ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())

