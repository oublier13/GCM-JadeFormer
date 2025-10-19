import os
import logging
from logging.handlers import RotatingFileHandler
import torch
import coloredlogs

def setup_logger(output_dir, name='train'):
    """
    配置日志系统，同时输出到文件和终端
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 防止重复添加handler
    if not logger.handlers:
        # 文件日志 (自动轮转)
        file_handler = RotatingFileHandler(
            filename=os.path.join(output_dir, 'training.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%m/%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # 控制台日志 (带颜色)
        console_handler = logging.StreamHandler()
        coloredlogs.install(
            level='INFO',
            logger=logger,
            fmt='%(asctime)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S'
        )
    
    # 添加PyTorch设备信息
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
    
    return logger

class MetricLogger:
    """训练指标记录器"""
    def __init__(self, logger, delimiter="\t"):
        self.logger = logger
        self.delimiter = delimiter
        self.meters = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def log(self, step=None, prefix=""):
        msg = []
        if step is not None:
            msg.append(f"Step: {step}")
        msg += [f"{k}: {v.avg:.4f}" for k, v in self.meters.items()]
        self.logger.info(self.delimiter.join(msg))

class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count