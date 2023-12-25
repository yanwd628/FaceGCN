# from model import common
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

device = "cuda" if torch.cuda.is_available() else "cpu"


# device = "cpu"

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Encoder(nn.Module):
    def __init__(self, in_feature, n_resblocks, n_filters):
        super(Encoder, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        conv = default_conv
        self.head = conv(in_feature, n_filters, kernel_size)
        self.encoder_1 = nn.Sequential(*[ResBlock(conv, n_filters,
                                                  kernel_size, act=act, res_scale=1) for _ in
                                         range(n_resblocks)])
        self.downsample_1 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=2, padding=1)

        self.encoder_2 = nn.Sequential(*[ResBlock(conv, n_filters,
                                                  kernel_size, act=act, res_scale=1) for _ in
                                         range(n_resblocks)])
        self.downsample_2 = nn.Conv2d(n_filters, n_filters * 2, kernel_size=kernel_size, stride=2, padding=1)

        self.encoder_3 = nn.Sequential(*[ResBlock(conv, n_filters * 2,
                                                  kernel_size, act=act, res_scale=1) for _ in
                                         range(n_resblocks)])

    def forward(self, x):
        res = self.head(x)
        res_enc1 = self.encoder_1(res)
        res_down1 = self.downsample_1(res_enc1)
        res_enc2 = self.encoder_2(res_down1)
        res_down2 = self.downsample_2(res_enc2)
        res_enc3 = self.encoder_3(res_down2)

        res_enc = [res_enc1, res_enc2, res_enc3]
        return res_enc


class Decoder(nn.Module):
    def __init__(self, out_feature, n_resblocks, n_filters):
        super(Decoder, self).__init__()
        kernel_size = 3
        stride = 2
        act = nn.ReLU(True)
        conv = default_conv
        self.decoder_3 = nn.Sequential(*[ResBlock(conv, n_filters * 2,
                                                  kernel_size, act=act, res_scale=1) for _ in
                                         range(n_resblocks)])
        self.upsample_2 = nn.Sequential(
            nn.Conv2d(n_filters * 2, n_filters * 4, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.decoder_2 = nn.Sequential(*[ResBlock(conv, n_filters,
                                                  kernel_size, act=act, res_scale=1) for _ in
                                         range(n_resblocks)])
        self.upsample_1 = nn.Sequential(
            nn.Conv2d(n_filters, n_filters * 4, 1, bias=False),
            nn.PixelShuffle(2)
        )
        # self.upsample_1 = nn.ConvTranspose2d(n_filters, n_filters, kernel_size=stride * 2, stride=stride, padding=1)
        self.decoder_1 = nn.Sequential(*[ResBlock(conv, n_filters,
                                                  kernel_size, act=act, res_scale=1) for _ in
                                         range(n_resblocks)])

        self.m_tail = conv(n_filters, out_feature, kernel_size)

    def forward(self, x, res_enc):
        res_dec3 = self.decoder_3(x + res_enc[-1])
        res_up2 = self.upsample_2(res_dec3)
        res_dec2 = self.decoder_2(res_up2 + res_enc[-2])
        res_up1 = self.upsample_1(res_dec2)
        res_dec1 = self.decoder_1(res_up1 + res_enc[-3])
        res_out = self.m_tail(res_dec1)

        return res_out


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight = nn.Parameter(torch.zeros((in_features, out_features)), requires_grad=True)

    def forward(self, inpu, adj):
        support = torch.matmul(inpu, self.weight)
        output = torch.matmul(adj, support)
        return output


class ResGCN(nn.Module):
    def __init__(self, features):
        super(ResGCN, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.graph_conv1 = GraphConvolution(features, features)
        self.graph_conv2 = GraphConvolution(features, features)

    def forward(self, ipt):
        x = ipt[0]
        adj = ipt[1]
        res_g = self.graph_conv1(x, adj)
        res_g = self.relu(res_g)
        res_g = self.graph_conv2(res_g, adj)
        return [x + res_g, adj]


class GraphConvModule(nn.Module):
    def __init__(self, n_graph_features, n_ResGCN):
        super(GraphConvModule, self).__init__()
        self.graph_convhead = GraphConvolution(1, n_graph_features)
        self.graph_convtail = GraphConvolution(n_graph_features, 1)
        self.GCN_body = nn.Sequential(*[ResGCN(n_graph_features) for _ in range(n_ResGCN)])
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        x = x.unsqueeze(3)
        x = self.graph_convhead(x, adj)
        x, adj_m = self.GCN_body([x, adj])
        x = self.graph_convtail(x, adj)
        x = self.relu(x)

        x = x.squeeze(3)
        return x


class IntraStripGCM(nn.Module):
    def __init__(self, n_features,
                 n_graph_features,
                 n_ResGCN):
        super(IntraStripGCM, self).__init__()
        self.norm = nn.LayerNorm(n_features)
        self.conv = default_conv(n_features, n_features, 1)
        self.fuse_out = default_conv(n_features, n_features, 1)
        self.graph_conv_h = GraphConvModule(n_graph_features, n_ResGCN)
        self.graph_conv_v = GraphConvModule(n_graph_features, n_ResGCN)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj_matrix):
        # feature -> graph -> graph conv -> graph -> feature
        # feature -> v and h -> each node is each one PIXEL in single strip

        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        x = self.norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        feature_h, feature_v = torch.chunk(self.conv(x), 2, dim=1)
        feature_h, feature_v = self.relu(feature_h), self.relu(feature_v)

        feature_h = feature_h.permute(0, 2, 1, 3).contiguous()
        feature_h = feature_h.view(B * H, C // 2, W)

        feature_v = feature_v.permute(0, 3, 1, 2).contiguous()
        feature_v = feature_v.view(B * W, C // 2, H)

        # for graph
        intra_adj_h, intra_adj_v = adj_matrix["intra_adj_h"], adj_matrix["intra_adj_v"]
        attention_output_h = self.graph_conv_h(feature_h, intra_adj_h)
        attention_output_v = self.graph_conv_v(feature_v, intra_adj_v)

        attention_output_h = attention_output_h.view(B, H, C // 2, W).permute(0, 2, 1, 3).contiguous()
        attention_output_v = attention_output_v.view(B, W, C // 2, H).permute(0, 2, 3, 1).contiguous()
        attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h

        return x


class InterStripGCM(nn.Module):
    def __init__(self, n_features,
                 n_graph_features,
                 n_ResGCN):
        super(InterStripGCM, self).__init__()
        self.norm = nn.LayerNorm(n_features)
        self.conv = default_conv(n_features, n_features, 1)
        self.fuse_out = default_conv(n_features, n_features, 1)
        self.graph_conv_h = GraphConvModule(n_graph_features, n_ResGCN)
        self.graph_conv_v = GraphConvModule(n_graph_features, n_ResGCN)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj_matrix):
        # feature -> graph -> graph conv -> graph -> feature
        # feature -> v and h -> each node is each single strip

        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        x = self.norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        feature_h, feature_v = torch.chunk(self.conv(x), 2, dim=1)
        feature_h, feature_v = self.relu(feature_h), self.relu(feature_v)

        feature_h = feature_h.permute(0, 1, 3, 2).contiguous()
        feature_h = feature_h.view(B, C // 2 * W, H)
        feature_v = feature_v.permute(0, 1, 2, 3).contiguous()
        feature_v = feature_v.view(B, C // 2 * H, W)

        # for graph
        inter_adj_h, inter_adj_v = adj_matrix["inter_adj_h"], adj_matrix["inter_adj_v"]
        attention_output_h = self.graph_conv_h(feature_h, inter_adj_h)
        attention_output_v = self.graph_conv_v(feature_v, inter_adj_v)

        attention_output_h = attention_output_h.view(B, C // 2, W, H).permute(0, 1, 3, 2).contiguous()
        attention_output_v = attention_output_v.view(B, C // 2, H, W).permute(0, 1, 2, 3).contiguous()
        attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h

        return x


class StripGCM(nn.Module):
    def __init__(self, in_feature,
                 out_feature,
                 n_features,
                 n_graph_features,
                 n_ResGCN):
        super(StripGCM, self).__init__()

        self.firstconv = default_conv(in_feature, out_feature, 1)
        self.Intra = IntraStripGCM(n_features, n_graph_features, n_ResGCN)
        self.Inter = InterStripGCM(n_features, n_graph_features, n_ResGCN)

    def forward(self, ipt):
        x = ipt[0]
        adj_matrix = ipt[1]
        res = self.firstconv(x)
        res = self.Intra(res, adj_matrix)
        res = self.Inter(res, adj_matrix)
        return [res, adj_matrix]


class DynamicAdj(nn.Module):
    def __init__(self, in_feature, n_filters, ipt_h, ipt_w, scale):
        super(DynamicAdj, self).__init__()
        n_resblocks = 2
        kernel_size = 3
        act = nn.ReLU(True)
        conv = default_conv
        self.first = default_conv(in_feature, n_filters, 1)
        self.share_body = nn.Sequential(*[ResBlock(conv, n_filters,
                                                   kernel_size, act=act, res_scale=1) for _ in
                                          range(n_resblocks)])
        self.share_downsample = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.intra_body = nn.Sequential(*[ResBlock(conv, n_filters,
                                                   kernel_size, act=act, res_scale=1) for _ in
                                          range(n_resblocks)])
        self.intra_downsample = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=scale // 2, padding=1)
        self.inter_body = nn.Sequential(*[ResBlock(conv, n_filters,
                                                   kernel_size, act=act, res_scale=1) for _ in
                                          range(n_resblocks)])
        self.inter_downsample = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=scale // 2, padding=1)
        self.relu = nn.ReLU(True)

        self.intra_v_head = nn.Conv2d(n_filters, 1, 1)
        self.intra_h_head = nn.Conv2d(n_filters, 1, 1)
        self.inter_v_head = nn.Conv2d(n_filters, 1, 1)
        self.inter_h_head = nn.Conv2d(n_filters, 1, 1)

        self.num_h_node = ipt_h // (4 * scale)
        self.num_w_node = ipt_w // (4 * scale)
        num_h_out = int(self.num_h_node * (self.num_h_node - 1) // 2)
        num_w_out = int(self.num_w_node * (self.num_w_node - 1) // 2)
        num_in = int(self.num_h_node * self.num_w_node)
        self.intra_v_body = nn.Linear(num_in, num_h_out)
        self.intra_h_body = nn.Linear(num_in, num_w_out)
        self.inter_v_body = nn.Linear(num_in, num_w_out)
        self.inter_h_body = nn.Linear(num_in, num_h_out)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.first(x)
        x = self.share_downsample(self.share_body(x))
        intra_feat = self.intra_downsample(self.relu(self.intra_body(x)))
        inter_feat = self.inter_downsample(self.relu(self.inter_body(x)))

        intra_half_adj_v = self.clip_act(self.intra_v_body(self.intra_v_head(intra_feat).view(b, -1)))
        intra_half_adj_h = self.clip_act(self.intra_h_body(self.intra_h_head(intra_feat).view(b, -1)))
        inter_half_adj_v = self.clip_act(self.inter_v_body(self.inter_v_head(inter_feat).view(b, -1)))
        inter_half_adj_h = self.clip_act(self.inter_h_body(self.inter_h_head(inter_feat).view(b, -1)))

        adj_matrix = []
        adj_ori_matrix = []
        for idx in range(b):
            intra_adj_v = self.construct_adj(intra_half_adj_v[idx], self.num_h_node)
            intra_adj_h = self.construct_adj(intra_half_adj_h[idx], self.num_w_node)
            inter_adj_v = self.construct_adj(inter_half_adj_v[idx], self.num_w_node)
            inter_adj_h = self.construct_adj(inter_half_adj_h[idx], self.num_h_node)
            adj_ori_matrix.append(intra_adj_v)
            adj_ori_matrix.append(intra_adj_h)
            adj_ori_matrix.append(inter_adj_v)
            adj_ori_matrix.append(inter_adj_h)
            adj_matrix_item = {
                "intra_adj_h": self.gen_adj(intra_adj_h),
                "intra_adj_v": self.gen_adj(intra_adj_v),
                "inter_adj_h": self.gen_adj(inter_adj_h),
                "inter_adj_v": self.gen_adj(inter_adj_v)
            }
            adj_matrix.append(adj_matrix_item)

        return adj_matrix, adj_ori_matrix

    def clip_act(self, x):
        return torch.where(x > 0, torch.tensor(1).to(device), torch.tensor(0).to(device))

    def construct_adj(self, half_adj, n):
        adj_matrix = torch.zeros(n, n)
        idx = 0
        for i in range(n):
            adj_matrix[i, i] = torch.tensor(1)
            for j in range(i + 1, n):
                adj_matrix[i, j] = half_adj[idx]
                adj_matrix[j, i] = half_adj[idx]
                idx += 1
        return adj_matrix

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D).to(device)
        return adj


class Encoder_latent(nn.Module):
    def __init__(self, in_feature, n_latent_resblocks, n_filters, scale):
        super(Encoder_latent, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        conv = default_conv
        self.head = conv(in_feature, n_filters, kernel_size)
        self.encoder_1 = nn.Sequential(*[ResBlock(conv, n_filters,
                                                  kernel_size, act=act, res_scale=1) for _ in
                                         range(n_latent_resblocks)])
        self.downsample_1 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=2, padding=1)

        self.encoder_2 = nn.Sequential(*[ResBlock(conv, n_filters,
                                                  kernel_size, act=act, res_scale=1) for _ in
                                         range(n_latent_resblocks)])
        self.downsample_2 = nn.Conv2d(n_filters, n_filters * 2, kernel_size=kernel_size, stride=2, padding=1)

        self.encoder_3 = nn.Sequential(*[ResBlock(conv, n_filters * 2,
                                                  kernel_size, act=act, res_scale=1) for _ in
                                         range(n_latent_resblocks)])
        self.downsample_3 = nn.Conv2d(n_filters * 2, n_filters * 2, kernel_size=kernel_size, stride=2, padding=1)

        self.encoder_4 = nn.Sequential(*[ResBlock(conv, n_filters * 2,
                                                  kernel_size, act=act, res_scale=1) for _ in
                                         range(n_latent_resblocks)])
        self.scale = scale
        assert self.scale in [4, 8]

    def forward(self, x):
        res = self.head(x)
        res_enc1 = self.encoder_1(res)
        res_down1 = self.downsample_1(res_enc1)
        res_enc2 = self.encoder_2(res_down1)
        res_down2 = self.downsample_2(res_enc2)
        res_enc3 = self.encoder_3(res_down2)
        if self.scale == 4:
            res_enc = [res_enc1, res_enc2, res_enc3]
        elif self.scale == 8:
            res_down3 = self.downsample_3(res_enc3)
            res_enc4 = self.encoder_4(res_down3)
            res_enc = [res_enc1, res_enc2, res_enc3, res_enc4]

        return res_enc


class Decoder_latent(nn.Module):
    def __init__(self, out_feature, n_latent_resblocks, n_filters, scale):
        super(Decoder_latent, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        conv = default_conv
        self.decoder_4 = nn.Sequential(*[ResBlock(conv, n_filters * 2,
                                                  kernel_size, act=act, res_scale=1) for _ in
                                         range(n_latent_resblocks)])
        self.upsample_3 = nn.Sequential(
            nn.Conv2d(n_filters * 2, n_filters * 8, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.decoder_3 = nn.Sequential(*[ResBlock(conv, n_filters * 2,
                                                  kernel_size, act=act, res_scale=1) for _ in
                                         range(n_latent_resblocks)])
        self.upsample_2 = nn.Sequential(
            nn.Conv2d(n_filters * 2, n_filters * 4, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.decoder_2 = nn.Sequential(*[ResBlock(conv, n_filters,
                                                  kernel_size, act=act, res_scale=1) for _ in
                                         range(n_latent_resblocks)])
        self.upsample_1 = nn.Sequential(
            nn.Conv2d(n_filters, n_filters * 4, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.decoder_1 = nn.Sequential(*[ResBlock(conv, n_filters,
                                                  kernel_size, act=act, res_scale=1) for _ in
                                         range(n_latent_resblocks)])
        self.m_tail = conv(n_filters, out_feature, kernel_size)
        self.scale = scale

    def forward(self, x, res_enc):
        idx = -1
        if self.scale == 4:
            res_up3 = x
        elif self.scale == 8:
            res_dec4 = self.decoder_4(x + res_enc[-1])
            res_up3 = self.upsample_3(res_dec4)
            idx = -2
        res_dec3 = self.decoder_3(res_up3 + res_enc[idx])
        res_up2 = self.upsample_2(res_dec3)
        res_dec2 = self.decoder_2(res_up2 + res_enc[idx - 1])
        res_up1 = self.upsample_1(res_dec2)
        res_dec1 = self.decoder_1(res_up1 + res_enc[idx - 2])
        res_out = self.m_tail(res_dec1)

        return res_out


@ARCH_REGISTRY.register()
class FaceGCN(nn.Module):
    def __init__(self, in_feature=3,
                 out_feature=3,
                 ipt_h=512,
                 ipt_w=512,
                 n_resblocks=8,
                 n_filters=64,
                 n_stripgcm=6,
                 scale_feat_graph=8,
                 n_graph_features=32,
                 n_ResGCN=2,
                 n_latent_resblocks=2):
        super(FaceGCN, self).__init__()
        self.encoder = Encoder(in_feature, n_resblocks, n_filters)

        self.downsample_graph = Encoder_latent(n_filters * 2, n_latent_resblocks, n_filters, scale_feat_graph)

        self.graph_conv = nn.Sequential(
            *[StripGCM(n_filters * 2, n_filters * 2, n_filters * 2, n_graph_features, n_ResGCN) for _ in
              range(n_stripgcm)])

        self.upsample_graph = Decoder_latent(n_filters * 2, n_latent_resblocks, n_filters, scale_feat_graph)

        self.decoder = Decoder(out_feature, n_resblocks, n_filters)

        self.dynamicadj = DynamicAdj(n_filters * 2, n_filters, ipt_h, ipt_w, scale_feat_graph)

    def forward(self, x, **kwargs):
        b = x.size()[0]

        res_enc = self.encoder(x)

        mid_latent = res_enc[-1]
        adj_matrix, adj_ori_matrix = self.dynamicadj(mid_latent)

        enc_latents = self.downsample_graph(mid_latent)

        latent = enc_latents[-1]
        samples = []
        for idx in range(b):
            sample_latent = latent[idx].unsqueeze(0)
            sample_adj = adj_matrix[idx]
            res_mid, _ = self.graph_conv([sample_latent, sample_adj])
            samples.append(res_mid)
        res_mid = torch.cat(samples, 0)

        res_mid = self.upsample_graph(res_mid, enc_latents)

        res_out = self.decoder(res_mid, res_enc)

        return res_out, adj_ori_matrix


if __name__ == '__main__':
    model = FaceGCN().to(device)
    ipt = torch.randn(2, 3, 512, 512).to(device)
    # res, res_adj = model(ipt)
    res, _ = model(ipt)
    print(res.shape)

    # import torch.onnx
    # model.eval()
    # torch.onnx.export(model, (ipt,), "./model.onnx")
