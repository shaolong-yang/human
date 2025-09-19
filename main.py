import logging
import subprocess
import time
import os
import signal
import sys
import psutil
import socket
import requests  

def is_port_in_use(port: int, host: str = 'localhost') -> bool:
    """检查端口是否被占用，增加主机参数"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0
    except Exception as e:
        logger.error(f"检查端口 {port} 时出错: {str(e)}")
        return False

def terminate_process(process):
    """终止进程及其所有子进程，跨平台兼容"""
    if not process or process.poll() is not None:
        return
    
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        
        for child in children:
            try:
                print(f"终止子进程 {child.pid}")
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        _, still_alive = psutil.wait_procs(children, timeout=5)
        for p in still_alive:
            print(f"强制杀死子进程 {p.pid}")
            p.kill()
        
        print(f"终止主进程 {parent.pid}")
        parent.terminate()
        try:
            parent.wait(timeout=5)
        except psutil.TimeoutExpired:
            print(f"强制杀死主进程 {parent.pid}")
            parent.kill()
            
    except psutil.NoSuchProcess:
        print(f"进程 {process.pid} 已不存在")
    except Exception as e:
        print(f"终止进程时出错: {str(e)}")

def start_tts_asr():
    """启动tts_asr服务"""
    print("正在启动tts_asr服务...")
    
    if sys.platform == "win32":
        process_kwargs = {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        }
    else:
        process_kwargs = {
            "preexec_fn": os.setsid,
            "start_new_session": True
        }
    
    try:
        tts_process = subprocess.Popen(
            ["python", "ttsasrStart.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            **process_kwargs
        )
    except Exception as e:
        print(f"创建tts_asr进程失败: {str(e)}")
        return None
    
    start_timeout = 300
    start_time = time.time()
    while time.time() - start_time < start_timeout:
        if tts_process.poll() is not None:
            print("tts_asr服务启动失败，输出:")
            print(tts_process.stdout.read())
            terminate_process(tts_process)
            return None
        
        try:
            output = tts_process.stdout.readline()
            if output:
                print(f"tts_asr: {output.strip()}")
                if "整合服务初始化完成" in output:
                    print("tts_asr服务启动成功")
                    return tts_process
        except Exception as e:
            print(f"读取tts_asr输出时出错: {str(e)}")
        
        time.sleep(0.5)
    
    print("tts_asr服务启动超时")
    terminate_process(tts_process)
    return None

def start_digital_human():
    """启动数字人项目，修复API文档访问问题"""
    print("正在启动数字人项目...")
    
    # 检查8000端口是否已被占用
    if is_port_in_use(8000):
        print("错误：8000端口已被占用，无法启动数字人服务")
        return None
    
    if sys.platform == "win32":
        process_kwargs = {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
        }
    else:
        process_kwargs = {
            "preexec_fn": os.setsid,
            "start_new_session": True
        }
    
    try:
        # 调整启动命令，确保正确启用文档
        human_process = subprocess.Popen(
            ["python", "-u", "humanStart.py"],  # 添加-u参数禁用输出缓冲
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,** process_kwargs
        )
    except Exception as e:
        print(f"创建数字人进程失败: {str(e)}")
        return None
    
    start_timeout = 300
    start_time = time.time()
    docs_available = False
    
    while time.time() - start_time < start_timeout:
        if human_process.poll() is not None:
            print("数字人项目启动失败，输出:")
            print(human_process.stdout.read())
            terminate_process(human_process)
            return None
        
        try:
            output = human_process.stdout.readline()
            if output:
                print(f"数字人: {output.strip()}")
                # 检查服务是否完全启动
                if "启动服务: http://" in output:
                    print("数字人服务已开始监听端口")
                # 检查文档是否可用
                if "Uvicorn running on" in output:
                    docs_available = True
                    print("API文档应该已可用: http://localhost:8000/docs")
                if "工作目录初始化完成" in output:
                    # 等待文档完全加载
                    time.sleep(2)
                    # 验证端口是否真的在监听
                    if is_port_in_use(8000):
                        print("数字人项目启动成功")
                        return human_process
                    else:
                        print("数字人服务未正确监听8000端口")
        except Exception as e:
            print(f"读取数字人输出时出错: {str(e)}")
        
        time.sleep(0.5)
    
    print("数字人项目启动超时")
    if not docs_available:
        print("警告：API文档可能未正确初始化")
    terminate_process(human_process)
    return None

# def check_digital_human_health():
#     """更全面的数字人服务健康检查"""
#     # 1. 检查端口是否监听
#     if not is_port_in_use(8000):
#         logger.warning("8000端口未监听，数字人服务可能已崩溃")
#         return False
    
#     # 2. 尝试访问健康检查接口
#     try:
#         response = requests.get("http://localhost:8000/health", timeout=5)
#         if response.status_code == 200:
#             logger.debug("数字人服务健康检查通过")
#             return True
#         else:
#             logger.warning(f"数字人服务健康检查失败，状态码: {response.status_code}")
#             return False
#     except requests.exceptions.RequestException as e:
#         logger.warning(f"无法连接到数字人服务健康检查接口: {str(e)}")
#         return False

def main():
    tts_process = None
    human_process = None
    # 增加日志器配置
    global logger
    logger = logging.getLogger("service_manager")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    try:
        # 检查Python版本
        if sys.version_info < (3, 7):
            logger.warning("Python版本低于3.7，可能存在兼容性问题")
        
        logger.info("开始启动tts_asr服务...")
        tts_process = start_tts_asr()
        if not tts_process:
            logger.error("启动tts_asr服务失败，无法继续")
            return
        
        time.sleep(5)
        
        logger.info("开始启动数字人项目...")
        human_process = start_digital_human()
        if not human_process:
            logger.error("启动数字人项目失败，终止tts_asr服务")
            terminate_process(tts_process)
            return
        
        logger.info("\n所有服务启动完成")
        logger.info("API文档地址: http://localhost:8000/docs")
        logger.info("按Ctrl+C停止所有服务...")
        
        # 连续失败计数器
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while True:
            if tts_process.poll() is not None:
                logger.error(f"tts_asr服务意外退出，退出码: {tts_process.returncode}")
                terminate_process(human_process)
                return
            
            if human_process.poll() is not None:
                logger.error(f"数字人项目意外退出，退出码: {human_process.returncode}")
                terminate_process(tts_process)
                return
            
            # 定期检查数字人服务健康状态
            # if not check_digital_human_health():
            #     consecutive_failures += 1
            #     logger.warning(f"数字人服务健康检查失败，连续失败次数: {consecutive_failures}/{max_consecutive_failures}")
                
            #     if consecutive_failures >= max_consecutive_failures:
            #         logger.error("数字人服务连续多次检查失败，尝试重启...")
            #         # 终止现有进程
            #         terminate_process(human_process)
            #         # 尝试重启
            #         human_process = start_digital_human()
            #         if not human_process:
            #             logger.error("重启数字人服务失败，终止所有服务")
            #             terminate_process(tts_process)
            #             return
            #         consecutive_failures = 0  # 重置计数器
            # else:
            #     consecutive_failures = 0  # 重置计数器
                
            # time.sleep(5)  # 延长检查间隔，减少资源消耗
    
    except KeyboardInterrupt:
        logger.info("\n收到停止信号，开始停止所有服务...")
    except Exception as e:
        logger.error(f"\n发生意外错误: {str(e)}", exc_info=True)
    finally:
        if human_process:
            print("正在停止数字人项目...")
            terminate_process(human_process)
        
        if tts_process:
            print("正在停止tts_asr服务...")
            terminate_process(tts_process)
        
        time.sleep(2)
        print("所有服务已停止")

if __name__ == "__main__":
    if sys.platform == "win32":
        import win32api
        def handle_win32_interrupt(sig, func):
            raise KeyboardInterrupt
        win32api.SetConsoleCtrlHandler(handle_win32_interrupt, True)
    
    main()
    