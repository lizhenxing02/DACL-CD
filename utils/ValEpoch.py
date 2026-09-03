from tqdm import tqdm
import torch
from torchnet.meter import AverageValueMeter
from utils.metrics import CMMeter
from utils.func import get_mem, list2device



class ValEpoch:
    '''
    封装验证阶段逻辑，计算评估指标（F1 分数、IoU 等），用于模型性能监控和最佳模型保存
    '''
    # 初始化方法，设置验证所需的参数和设备
    def __init__(self, num_classes, net,
                criterion1,
                metric, device="cuda"):

        self.num_classes = num_classes # 类别数
        self.net = net # 模型网络
        self.criterion = criterion1  # 损失函数
        # self.criterion_consistency = criterion_consistency
        self.metric = metric  # 评估指标计算方法
        self.device = device  # 使用的设备，默认为CUDA

        self._to_device()  # 将网络、损失函数和指标计算方法移动到指定设备

    # 将网络、损失函数和指标计算方法移动到指定设备的方法
    def _to_device(self):
        self.net.to(self.device)
        self.criterion.to(self.device)
        self.metric.to(self.device)

    # 在验证集上运行一个轮次的方法
    @torch.no_grad() # 不计算梯度，因为验证过程中不需要更新模型参数
    def run(self, dataloader):
        # 打印验证过程的表头
        print(('\n' + '%10s' * 8) % ("val", 'gpu', 'loss', 'precision', 'recall', 'f1', 'iou', 'OA'))

        # # 将模型设置为测试模式，以禁用Dropout等仅在训练时需要的操作
        self.net.eval()

        # # 初始化损失loss和指标的记录器
        loss_meter = AverageValueMeter()
        cm_meter = CMMeter()

        # 创建进度条
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, sample in pbar:
        # for step, (sample, _) in pbar:
            # x = x.to(self.device)
            # 将输入数据移动到指定设备
            x = list2device(sample['image'], self.device)
            label = sample['labels'].to(self.device)
            Chg = self.net(x)

            # 计算损失
            loss = self.criterion(Chg, label)
            loss_labeled_value = loss.cpu().detach().numpy()
            loss_meter.add(loss_labeled_value)
            # 计算指标
            metrics = self.metric(Chg, label)
            cm_meter.add(metrics)
            precision, recall, f1, iou, oa = cm_meter.get_metrics()
            
            # 更新进度条
            pbar.set_description(('%10s' * 2 + '%10.4g' * 6) % ("val", get_mem(), loss_labeled_value,
                                                                precision, recall, f1, iou, oa))
        # 最终计算平均指标
        precision, recall, f1, iou, oa = cm_meter.get_metrics()
        logs = {
            'loss': loss_meter.mean,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'oa': oa
        }
        return logs