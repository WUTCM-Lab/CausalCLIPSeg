import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip import build_model

from .layers import Projector, FPN_AD

def conv_bn(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim))


class CausalCLIPSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
        self.neck_ad = FPN_AD(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)
        self.proj_ad = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)

    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        vis = self.backbone.encode_image(img)
        word, state = self.backbone.encode_text(word)

        
        #ad_decoder
        fq_sup,fq_inf= self.neck_ad(vis, state)


        pred = self.proj(fq_sup, state)
        pred_ad = self.proj_ad(fq_inf, state)
        
        return pred, pred_ad