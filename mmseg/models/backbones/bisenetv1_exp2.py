# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule

from ..builder import BACKBONES, build_backbone
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
                 out_indices=(0, 1, 2),
                 **kwargs):

        super(BiSeNetV1EXPCFG, self).__init__(init_cfg)

        self.context_path = build_context_path(context_path_cfg)
        self.spatial_path = build_spatial_path(spatial_path_cfg)
        self.ffm = build_ffm(ffm_cfg)
        self.out_indices = out_indices

    def forward(self, x):
        # stole refactoring code from Coin Cheung, thanks
        # x_8, x_16, x_32 = self.context_path(x)
        x_16, x_32 = self.context_path(x)

        x_spatial = self.spatial_path(x)
        x_fuse = self.ffm(x_spatial, x_32)
        outs = [x_fuse, x_16]
        # outs = [outs[i] for i in self.out_indices]
        return tuple(outs)


@BACKBONES.register_module()
class BiSeNetSWCFG(BaseModule):
    """BiSeNetV1 backbone.
    """

    def __init__(self,
                 backbone_cfg,
                 ffm_cfg,
                 init_cfg=None,
                 out_indices=(0, 1, 2),
                 **kwargs):

        super(BiSeNetSWCFG, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone_cfg)

        self.ffm = build_ffm(ffm_cfg)
        self.out_indices = out_indices

    def forward(self, x):
        x_8, x_32 = self.backbone(x)
        x_fuse = self.ffm(x_8, x_32)
        return [x_8, x_32, x_fuse]


@BACKBONES.register_module()
class BiSeNetSWContext(BaseModule):
    """BiSeNetV1 backbone.
    """

    def __init__(self,
                 context_path_cfg,
                 ffm_cfg,
                 init_cfg=None,
                 out_indices=(0, 1, 2),
                 **kwargs):

        super(BiSeNetSWContext, self).__init__(init_cfg)

        self.backbone = build_context_path(context_path_cfg)

        self.ffm = build_ffm(ffm_cfg)
        self.out_indices = out_indices

    def forward(self, x):
        x_spatial_8, x_context_8, x_context_16 = self.backbone(x)
        x_fuse = self.ffm(x_spatial_8, x_context_8)
        return [x_spatial_8, x_context_16, x_fuse]


@BACKBONES.register_module()
class BiSeNetSimpleContext(BaseModule):
    """BiSeNetV1 backbone.
    """

    def __init__(self,
                 context_path_cfg,
                 ffm_cfg,
                 init_cfg=None,
                 out_indices=(0, 1, 2),
                 **kwargs):

        super(BiSeNetSimpleContext, self).__init__(init_cfg)

        self.backbone = build_context_path(context_path_cfg)

        self.ffm = build_ffm(ffm_cfg)
        self.out_indices = out_indices

    def forward(self, x):
        x_8, x_16, x_32 = self.backbone(x)
        x_fuse = self.ffm(x_8, x_32)
        return [x_8, x_16, x_fuse]


@BACKBONES.register_module()
class SimpleBiSeNet(BaseModule):
    def __init__(
            self,
            backbone_cfg,
            spatial_path_cfg,
            init_cfg=None,
            **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone_cfg)
        self.spatial_path = build_spatial_path(spatial_path_cfg)

    def forward(self, img):
        # x32
        context_path = self.backbone(img)
        # x8
        spatial_path = self.spatial_path(img)
        return tuple([spatial_path, context_path[0]])
