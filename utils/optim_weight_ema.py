import torch

class EMAWeightOptimizer(object):
    """
    实现 EMA 优化器，更新教师网络参数
    指数移动平均（Exponential Moving Average, EMA）权重优化器类。
    该类用于在训练过程中更新目标网络的权重，使其逐渐接近源网络的权重。
    """
    def __init__(self, target_net, source_net, ema_alpha):
        """
        初始化EMAWeightOptimizer对象。

        参数：
        - target_net: 目标网络，其权重将被更新。
        - source_net: 源网络，其权重将作为参考来更新目标网络。
        - ema_alpha: EMA平滑因子，值越接近1，更新越慢。
        """
        self.target_net = target_net  # 目标网络
        self.source_net = source_net  # 源网络
        self.ema_alpha = ema_alpha  # EMA平滑因子

        # 提取目标网络和源网络中所有浮点类型的参数
        self.target_params = [p for p in target_net.state_dict().values() if p.dtype == torch.float]
        self.source_params = [p for p in source_net.state_dict().values() if p.dtype == torch.float]

        # 将目标网络的初始权重设置为与源网络相同
        for tgt_p, src_p in zip(self.target_params, self.source_params):
            tgt_p[...] = src_p[...]

        # 获取目标网络和源网络的状态字典键集合
        target_keys = set(target_net.state_dict().keys())
        source_keys = set(source_net.state_dict().keys())

        # 如果目标网络和源网络的状态字典键不一致，则抛出异常
        if target_keys != source_keys:
            raise ValueError('源网络和目标网络的状态字典键不一致；它们是否具有不同的架构？')

    def step(self):
        """
        执行一步EMA权重更新操作。
        更新规则：target_param = ema_alpha * target_param + (1 - ema_alpha) * source_param
        """
        one_minus_alpha = 1.0 - self.ema_alpha  # 计算1 - ema_alpha

        # 遍历目标网络和源网络的参数，按EMA公式更新目标网络的权重
        for tgt_p, src_p in zip(self.target_params, self.source_params):
            tgt_p.mul_(self.ema_alpha)  # 目标参数乘以ema_alpha
            tgt_p.add_(src_p * one_minus_alpha)  # 加上源参数乘以(1 - ema_alpha)