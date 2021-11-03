# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule

from ..builder import BACKBONES
from ..utils import build_spatial_path, build_ffm, build_context_path


@BACKBONES.register_module()
class BiSeNetV1EXPCFG(BaseModule):
    """BiSeNetV1 backbone.
    """

    def __init__(self,
                 context_path_cfg,
                 spatial_path_cfg,
                 ffm_cfg,
                 init_cfg=None,
                 **kwargs):

        super(BiSeNetV1EXPCFG, self).__init__(init_cfg)

        self.context_path = build_context_path(context_path_cfg)
        self.spatial_path = build_spatial_path(spatial_path_cfg)
        self.ffm = build_ffm(ffm_cfg)

    def forward(self, x):
        # stole refactoring code from Coin Cheung, thanks
        x_8, x_16, x_32 = self.context_path(x)
        x_spatial = self.spatial_path(x)
        x_fuse = self.ffm(x_spatial, x_32)

        outs = [x_fuse, x_8, x_16]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
