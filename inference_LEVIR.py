import os
import argparse
from utils.func import Logger, format_logs
from utils.metrics import *
import torch
import torch.nn as nn
import os
import time
from torch.utils.data.dataloader import DataLoader
import models
from loaders.datasets import LEVIRDataset
from tqdm import tqdm
from utils.ValEpoch import ValEpoch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_arguments():
    """Parse all the arguments provided from the CLI.解析命令行参数

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="train process")
    parser.add_argument("--work_dirs", type=str, default='./semi_checkpoints/LEVIR/CutMixCD')
    parser.add_argument("--log", type=str, default='cutmix_0.5')
    parser.add_argument("--weight_path", type=str, default='./semi_checkpoints/LEVIR/CutMixCD/cutmix_0.5/best.pth')
    parser.add_argument("--data_root", type=str, default='./LEVIR-CD256')
    parser.add_argument("--test-batch-size", type=int, default=16,
                        help="test dataset batch size.")

    return parser.parse_args()


def test():
    num_classes = 2
    torch_device = torch.device('cuda')
    args = get_arguments() # 调用get_arguments函数获取命令行参数

    # logger日志
    save_dir = os.path.join(args.work_dirs, args.log) # 定义保存日志的目录路径
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，则创建该目录（exist_ok=True表示如果目录已存在则不报错）
    logger = Logger(os.path.join(save_dir, "test.log")) # 初始化Logger对象，用于记录日志信息
    logger.write(str(args)) # 将解析的参数写入日志文件

    # test dataset测试数据集
    test_dataset = LEVIRDataset(args.data_root, "test") # 初始化LEVIRDataset对象，加载测试数据集
    # 使用DataLoader加载测试数据集，设置批量大小、工作线程数等参数
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=args.test_batch_size,
                            num_workers=1,
                            shuffle=False,
                            pin_memory=True)

    # load model加载模型
    # 初始化EfficientNetB0_CD模型，并将其移动到CUDA设备上
    eval_net = models.EfficientNetB0_CD(num_classes, pretrained=None).to(torch_device)
    checkpoint = torch.load(args.weight_path) # 加载预训练模型权重
    eval_net.load_state_dict(checkpoint['net'])  # 将加载的权重应用到模型中
    _ = eval_net.eval() # 将模型设置为评估模式

    CE_loss = nn.CrossEntropyLoss() # 定义交叉熵损失函数

    # test测试
    metric = ChangeMetrics(False) # 初始化ChangeMetrics对象，用于计算变化检测指标
    test_runner = ValEpoch(num_classes, eval_net, CE_loss, metric) # 初始化ValEpoch对象，用于执行测试过程

    # Eval this epoch
    test_log = test_runner.run(test_loader)
    val_metric = test_log['f1'] # 获取F1分数

    logger.write('Test:\t' + format_logs(test_log))  # 将测试结果写入日志文件
    print("Test:", test_log)

if __name__ == "__main__":
    test()