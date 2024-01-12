import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from torch.nn.functional import grid_sample
from pytorch3d.renderer.cameras import PerspectiveCameras
from nnutils import geom_utils, mesh_utils, slurm_utils
from nnutils.hand_utils import ManopthWrapper
import random

def get_embedder(multires=10, **kwargs):
    if multires == -1:
        return nn.Identity(), kwargs.get('input_dims', 3)

    embed_kwargs = {
        'include_input': kwargs.get('include_input', True),
        'input_dims': kwargs.get('input_dims', 3),
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': kwargs.get('log_sampling', True),
        'band_width': kwargs.get('band_width', 1),
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    # embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embedder_obj, embedder_obj.out_dim

def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.modules.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.modules.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.modules.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())

class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(16, 16, 'subm0')
        self.down0 = stride_conv(16, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x):
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()

        volumes = [net1, net2, net3, net4]

        return volumes



# Positional encoding (section 5.1)
class Embedder:
    """from https://github.com/yenchenlin/nerf-pytorch/blob/bdb012ee7a217bfd96be23060a51df7de402591e/run_nerf_helpers.py"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims'] # 3
        band_width = self.kwargs['band_width']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = band_width * 2. ** torch.linspace(0., max_freq, steps=N_freqs) # tensor([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.])
        else:
            freq_bands = torch.linspace(band_width * 2. ** 0., band_width * 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

    def __call__(self, inputs):
        return self.embed(inputs)


class PixCoord(nn.Module):
    def __init__(self, cfg, z_dim, hA_dim, freq):
        super().__init__()
        J = 16
        self.holistic_net = ImplicitNetwork(304, multires=freq, **cfg.SDF) # 304 10
        self.downsampling = torch.nn.Linear(656, 304)
        self.xyzc_net = SparseConvNet()

    def get_dist_joint(self, nPoints, jsTn):
        N, P, _ = nPoints.size()
        num_j = jsTn.size(1)
        nPoints_exp = nPoints.view(N, 1, P, 3).expand(N, num_j, P, 3).reshape(N * num_j, P, 3)
        jsPoints = mesh_utils.apply_transform(nPoints_exp, jsTn.reshape(N*num_j, 4, 4)).view(N, num_j, P, 3)
        jsPoints = jsPoints.transpose(1, 2).reshape(N, P, num_j * 3) # N, P, J, 3
        return jsPoints  # (N, P, J)

    def sample_multi_z(self, xPoints, z, cTx, cam):
        N1, P, D = xPoints.size()
        N = z.size(0)
        xPoints_exp = xPoints.expand(N, P, D) # torch.Size([2, 24000, 3])
        ndcPoints = self.proj_x_ndc(xPoints_exp, cTx, cam) # torch.Size([2, 24000, 2])
        zs = mesh_utils.sample_images_at_mc_locs(z, ndcPoints)  # (N, P, D)
        return zs

    def proj_x_ndc(self, xPoints, cTx, cam:PerspectiveCameras):
        cPoints = mesh_utils.apply_transform(xPoints, cTx)
        ndcPoints = mesh_utils.transform_points(cPoints, cam)
        return ndcPoints[..., :2]

    def forward(self, xPoints, z, hA, contact_feat=torch.FloatTensor([0] * 778).unsqueeze(0), xHand=None, cTx=None, cam: PerspectiveCameras=None, jsTx=None):
        N, P, _ = xPoints.size() # torch.Size([2, 24000, 3]) for training
        glb, local = z # torch.Size([2, 256]) torch.Size([2, 256, 56, 56])
        local = self.sample_multi_z(xPoints, local, cTx, cam) # torch.Size([2, 24000, 256])
        dstPoints = self.get_dist_joint(xPoints, jsTx) # torch.Size([2, 24000, 48])
        contact_feat_spc = contact_feat.permute(1,0) # torch.Size([778, 2])
        latent = self.cat_z_hA((glb, local, dstPoints), hA) # torch.Size([2, 24000, 304])

        contact_feat_spc_list = []
        for i in range(contact_feat_spc.size(1)):
            contact_feat_spc_list.append(contact_feat_spc[:, i].unsqueeze(1))
        code = torch.cat(contact_feat_spc_list, 0)  # torch.Size([1556, 1])
        embed_fn, input_ch = get_embedder(multires=8, input_dims=1, include_input=False)  # 10, 3
        contact_code = embed_fn(code)  # torch.Size([1556, 16])

        hand_verts = xHand.verts_list()  # torch.Size([778, 3]) in a batch length list
        hand_coord = torch.zeros([778 * N, 4]).to(latent.device)
        for i in range(len(hand_verts)):
            single_hand_verts = hand_verts[i]
            min_xyz = -1.0
            max_xyz = 1.0
            single_hand_coord = 63 * (single_hand_verts - min_xyz) / (max_xyz - min_xyz)
            hand_coord[i * 778:(i + 1) * 778, 0] = i
            hand_coord[i * 778:(i + 1) * 778, 1:] = single_hand_coord[:, [2, 1, 0]]
        hand_coord = torch.as_tensor(hand_coord, dtype=torch.int32)
        out_sh = [64, 64, 64]
        xyzc = spconv.SparseConvTensor(contact_code, hand_coord, out_sh, N)  # code:contact label(778*1)->(778*16) by using positional encoding of nerf to increase the coordinate sensitivity

        # add sparseconvnet operation
        feature_volume = self.xyzc_net(xyzc)
        xyzc_dense0 = feature_volume[0] # torch.Size([2, 32, 32, 32, 32])
        xyzc_dense1 = feature_volume[1]  # torch.Size([2, 64, 16, 16, 16])
        xyzc_dense2 = feature_volume[2]  # torch.Size([2, 128, 8, 8, 8])
        xyzc_dense3 = feature_volume[3]  # torch.Size([2, 128, 4, 4, 4])
        obj_verts = xPoints[:, :, :3]  # torch.Size([2, 24000, 3])
        min_xyz = -1.0
        max_xyz = 1.0
        obj_coord0 = 63 * (obj_verts - min_xyz) / (max_xyz - min_xyz)
        obj_coord1 = torch.as_tensor(obj_coord0, dtype=torch.int32)
        min_xyz = 0.0
        max_xyz = 63.0
        obj_coord2 = 2 * (obj_coord1 - min_xyz) / (max_xyz - min_xyz) - 1
        obj_coord2 = obj_coord2.view(N, P, 1, 1, 3) # torch.Size([2, 24000, 3])
        sp_feat0 = grid_sample(xyzc_dense0,obj_coord2,mode='nearest',align_corners=True)[..., 0, 0] #point_cloud [-1,1] 1-NN # torch.Size([2, 32, 24000])
        sp_feat1 = grid_sample(xyzc_dense1, obj_coord2, mode='nearest', align_corners=True)[..., 0, 0]  # point_cloud [-1,1] 1-NN # torch.Size([2, 32, 24000])
        sp_feat2 = grid_sample(xyzc_dense2, obj_coord2, mode='nearest', align_corners=True)[..., 0, 0]  # point_cloud [-1,1] 1-NN # torch.Size([2, 32, 24000])
        sp_feat3 = grid_sample(xyzc_dense3, obj_coord2, mode='nearest', align_corners=True)[..., 0, 0]  # point_cloud [-1,1] 1-NN # torch.Size([2, 32, 24000])
        sp_feat_cat = torch.cat([sp_feat0, sp_feat1, sp_feat2, sp_feat3], dim=1).permute(0,2,1)  # torch.Size([2, 24000, 352])  # torch.Size([2, 16384, 352])
        # todo: add loop opera, contact_feat -> field feature
        new_latent = torch.cat([latent, sp_feat_cat], dim=-1) # torch.Size([2, 24000, 656])
        new_latent = self.downsampling(new_latent) # torch.Size([2, 24000, 304])
        points = self.holistic_net.cat_z_point(xPoints, new_latent)  # torch.Size([48000, 659])
        sdf_value = self.holistic_net(points)  # torch.Size([48000, 1])
        sdf_value = sdf_value.view(N, P, 1)  # torch.Size([2, 24000, 1]) 0.9768, -0.1311

        return sdf_value

    def gradient(self, xPoints, sdf):
        """
        Args:
            x ([type]): (N, P, 3)
        Returns:
            Grad sdf_x: (N, P, 3)
        """
        xPoints.requires_grad_(True)
        y = sdf(xPoints)  # (N, P, 1)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=xPoints,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        # only care about xyz dim
        # gradients = gradients[..., -3:]

        # only care about sdf within cube
        xyz = xPoints[..., -3:]
        within_cube = torch.all(torch.abs(xyz) < 1, dim=-1, keepdim=True).float()  # (NP, )
        gradients = within_cube * gradients + (1 - within_cube) * 1 / np.sqrt(gradients.size(-1))

        if self.cfg.GRAD == 'clip':
            mask = (y.abs() <= 0.1).float()
            gradients = mask * gradients
        else:
            pass
        return gradients

    def cat_z_hA(self, z, hA):
        glb, local, dst_points = z
        out = torch.cat([(glb.unsqueeze(1) + local), dst_points], -1)
        return out


class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            latent_dim,
            feature_vector_size=0,
            d_in=3,
            d_out=1,
            DIMS=[ 512, 512, 512, 512, 512, 512, 512, 512 ], 
            GEOMETRIC_INIT=False,
            bias=1.0,
            SKIP_IN=(4, ),
            weight_norm=True,
            multires=10,
            th=True,
            **kwargs
    ):
        self.xyz_dim = d_in
        super().__init__()
        dims = [d_in + latent_dim] + list(DIMS) + [d_out + feature_vector_size] #[307, 512, 512, 512, 512, 512, 512, 512, 512, 1]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in) # 10, 3
            self.embed_fn = embed_fn
            dims[0] = input_ch + latent_dim

        self.num_layers = len(dims)
        self.skip_in = SKIP_IN
        self.layers = nn.ModuleDict()
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if GEOMETRIC_INIT:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    # torch.nn.init.constant_(lin.weight[:, 0:latent_dim], 0.0)
                    torch.nn.init.constant_(lin.weight[:, latent_dim+3:], 0.0)
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            # setattr(self, "lin" + str(l), lin)
            self.layers.add_module("lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        if th:
            self.th = nn.Tanh()
        else:
            self.th = nn.Identity()
        
    def forward(self, input, compute_grad=False):
        xyz = input[:, -self.xyz_dim:] # torch.Size([48000, 3])
        latent = input[:, :-self.xyz_dim] # torch.Size([48000, 304])
        if self.embed_fn is not None:
            xyz = self.embed_fn(xyz) # torch.Size([48000, 63])
        input = torch.cat([latent, xyz], dim=1) # torch.Size([48000, 367])
        x = input

        for l in range(0, self.num_layers - 1):
            lin = self.layers["lin" + str(l)]

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
        
        x = self.th(x)
        within_cube = torch.all(torch.abs(xyz) <= 1, dim=-1, keepdim=True).float()  # (NP, )
        apprx_dist= .3
        x = within_cube * x + (1 - within_cube) * (apprx_dist) # torch.Size([48000, 1])
        return x

    def gradient(self, x, sdf=None):
        """
        :param x: (sumP, D?+3)
        :return: (sumP, 1, 3)
        """
        x.requires_grad_(True)
        if sdf is None:
            y = self.forward(x)[:, :1]
        else:
            y = sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        # only care about xyz dim
        gradients = gradients[..., -3:]

        # only care about sdf within cube
        xyz = x[..., -self.xyz_dim:]
        within_cube = torch.all(torch.abs(xyz) < 1, dim=-1, keepdim=True).float()  # (NP, )
        gradients = within_cube * gradients + (1 - within_cube) * 1 / np.sqrt(gradients.size(-1))
        return gradients.unsqueeze(1)

    def cat_z_point(self, points, z):
        """
        :param points: (N, P, 3)
        :param z: (N, (P, ), D)
        :return: (NP, D+3)
        """
        if z.ndim == 3:
            N, P, D = z.size()
            return torch.cat([z, points], dim=-1).reshape(N*P, D+3)
        N, D = z.size()
        if points.ndim == 2:
            points = points.unsqueeze(0)
        NP, P, _ = points.size()
        assert N == NP

        z_p = torch.cat([z.unsqueeze(1).repeat(1, P, 1), points], dim=-1)
        z_p = z_p.reshape(N * P, D + 3)
        return z_p


class PixObj(PixCoord):
    def __init__(self, cfg, z_dim, hA_dim, freq):
        super().__init__(cfg, z_dim, hA_dim, freq)
        J = 16
        self.net = ImplicitNetwork(z_dim, multires=freq, 
            **cfg.SDF)
    
    def cat_z_hA(self, z, hA):
        glb, local, _ = z 
        glb = glb.unsqueeze(1)
        return glb + local


def build_net(cfg, z_dim=None):
    if z_dim is None:
        z_dim = cfg.Z_DIM
    if cfg.DEC == 'obj':
        dec = PixObj(cfg, z_dim, cfg.THETA_DIM, cfg.FREQ)
    else:
        dec = PixCoord(cfg, z_dim, cfg.THETA_DIM, cfg.FREQ) # z_dim=256, THETA_DIM=45, FREQ=10
    return dec

