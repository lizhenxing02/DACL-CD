import numpy as np
from torch.utils.data._utils.collate import default_collate
# 掩码生成器框架

class MaskGenerator (object):
    """
    Mask Generator掩码生成器基类，定义了掩码生成的基本接口
    """

    def generate_params(self, n_masks, mask_shape, rng=None): # 生成掩码参数
        raise NotImplementedError('Abstract')

    def append_to_batch(self, *batch): # 将生成的掩码参数附加到批次数据中
        x = batch[0]
        params = self.generate_params(len(x), x.shape[2:4])
        return batch + (params,)

    def torch_masks_from_params(self, t_params, mask_shape, torch_device): # 将掩码参数转换为 PyTorch 张量
        raise NotImplementedError('Abstract')


def gaussian_kernels(sigma, max_sigma=None, truncate=4.0):
    """
    Generate multiple 1D gaussian convolution kernels

    sigma: values for sigma as a `(N,)` array  作为‘ (N,) ’数组的sigma值
    max_sigma: maximum possible value for sigma or None to compute it; used to compute kernel size   sigma 的最大值或 None，用于计算核大小
    truncate: kernel size truncation factor  核大小截断因子
    :return: kernels as a `(N, kernel_size)` array
    """
    if max_sigma is None:
        max_sigma = sigma.max()
    sigma = sigma[:, None]
    radius = int(truncate * max_sigma + 0.5) # 计算高斯核的半径 radius
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)[None, :]
    phi_x = np.exp(-0.5 / sigma2 * x ** 2) # 计算高斯核
    phi_x = phi_x / phi_x.sum(axis=1, keepdims=True) # 归一化高斯核，使其总和为 1
    return phi_x


class BoxMaskGenerator (MaskGenerator): # 继承自: MaskGenerator，实现了具体的掩码生成逻辑
    def __init__(self, prop_range, n_boxes=1, random_aspect_ratio=True, prop_by_area=True, within_bounds=True, invert=False):
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_area = prop_by_area
        self.within_bounds = within_bounds
        self.invert = invert

    def generate_params(self, n_masks, mask_shape, rng=None):
        """
        生成方框掩码参数
        Box masks can be generated quickly on the CPU so do it there.

        >>> boxmix_gen = BoxMaskGenerator((0.25, 0.25))
        >>> params = boxmix_gen.generate_params(256, (32, 32))
        >>> t_masks = boxmix_gen.torch_masks_from_params(params, (32, 32), 'cuda:0')

        :param n_masks: number of masks to generate (batch size) 要生成的掩码数量（批大小）
        :param mask_shape: Mask shape as a `(height, width)` tuple 掩码形状为' (height, width) '元组
        :param rng: [optional] np.random.RandomState instance [可选]np.random.RandomState实例
        :return: masks: masks as a `(N, 1, H, W)` array 掩码为‘ (N, 1， H， W) ’数组
        """
        if rng is None:
            rng = np.random

        if self.prop_by_area:
            # Choose the proportion of each mask that should be above the threshold
            # 选择每个掩码的比例（应该高于阈值）
            mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))

            # Zeros will cause NaNs, so detect and suppres them
            # 零会产生纳米网络，所以要检测并抑制它们
            zero_mask = mask_props == 0.0

            if self.random_aspect_ratio: # 框的形状
                y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))
                x_props = mask_props / y_props
            else:
                y_props = x_props = np.sqrt(mask_props)  # 开根号  正方形
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

            y_props[zero_mask] = 0
            x_props[zero_mask] = 0
        else:
            if self.random_aspect_ratio:
                y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                x_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            else:
                x_props = y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

        sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])

        if self.within_bounds:   # mask不超出图像边界
            positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:   # mask可以超出图像边界
            centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        if self.invert: # 根据 invert 参数，初始化掩码为全零或全一
            masks = np.zeros((n_masks, 1) + mask_shape)
        else:
            masks = np.ones((n_masks, 1) + mask_shape)

        for i, sample_rectangles in enumerate(rectangles): # 遍历所有方框，填充掩码
            for y0, x0, y1, x1 in sample_rectangles:
                masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(y0):int(y1), int(x0):int(x1)]
        return masks

    #  直接返回输入的掩码参数（未做任何转换）
    def torch_masks_from_params(self, t_params, mask_shape, torch_device):
        return t_params

# boxmix_gen = BoxMaskGenerator((0.25, 0.25),random_aspect_ratio=False)
# params = boxmix_gen.generate_params(1, (8, 8))
# t_masks = boxmix_gen.torch_masks_from_params(params, (8, 8), 'cpu')
# print(t_masks)
# image = np.random.randint(1,10, size=[1,3,8,8])
# print(image)
# out = image * t_masks
# print(out)





class AddMaskParamsToBatch (object):
    """
    将掩码参数添加到批次数据中
    We add the cut-and-paste parameters to the mini-batch within the collate function,
    (we pass it as the `batch_aug_fn` parameter to the `SegCollate` constructor)
    as the collate function pads all samples to a common size
    """
    def __init__(self, mask_gen): # 初始化时传入一个掩码生成器实例
        self.mask_gen = mask_gen

    def __call__(self, batch):
        sample = batch[0]
        if 'sample0' in sample:
            sample0 = sample['sample0']
        else:
            sample0 = sample

        mask_size = sample0['image'][0].shape[1:3]
        params = self.mask_gen.generate_params(len(batch), mask_size) # 生成掩码参数
        for sample, p in zip(batch, params): # 将掩码参数附加到每个样本中
            sample['mask'] = p.astype(np.float32)
        return default_collate(batch) # 使用 default_collate 将样本组合成一个批次