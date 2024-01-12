"""
This script is borrowed and extended from https://github.com/JudyYe/ihoi
"""
import functools
from typing import Any, List
import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch.optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch3d.renderer.cameras import PerspectiveCameras
from config.args_config import default_argument_parser, setup_cfg
from nnutils.logger import MyLogger
from datasets import build_dataloader
from nnutils.hand_utils import ManopthWrapper
from nnutils import geom_utils, mesh_utils, slurm_utils
from models import enc
import models.dec_spconv as dec

def get_hTx(frame, batch):
    hTn = geom_utils.inverse_rt(batch['nTh'])
    hTx = hTn
    return hTx


def get_jsTx(hand_wrapper, hA, hTx):
    """
    Args:
        hand_wrapper ([type]): [description]
        hA ([type]): [description]
        hTx ([type]): se3
    Returns: 
        (N, 4, 4)
    """
    hTjs = hand_wrapper.pose_to_transform(hA, False) 
    N, num_j, _, _ = hTjs.size()
    jsTh = geom_utils.inverse_rt(mat=hTjs, return_mat=True)
    hTx = geom_utils.se3_to_matrix(hTx
            ).unsqueeze(1).repeat(1, num_j, 1, 1)
    jsTx = jsTh @ hTx
    return jsTx



class CHOI(pl.LightningModule):
    def __init__(self, cfg, **kwargs) -> None:
        super().__init__()
        self.hparams.update(cfg)
        self.cfg = cfg

        self.dec = dec.build_net(cfg.MODEL)  # sdf
        self.enc = enc.build_net(cfg.MODEL.ENC, cfg)
        self.hand_wrapper = ManopthWrapper()

        self.minT = -cfg.LOSS.SDF_MINMAX # 0.1
        self.maxT = cfg.LOSS.SDF_MINMAX
        self.sdf_key = '%sSdf' % cfg.MODEL.FRAME[0]
        self.obj_key = '%sObj' % cfg.MODEL.FRAME[0]
        self.metric = 'val'
        self._train_loader = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.SOLVER.BASE_LR)

    def train_dataloader(self):
        if self._train_loader is None:
            loader = build_dataloader(self.cfg, 'train')
            self._train_loader = loader
        return self._train_loader

    def val_dataloader(self):
        test = self.cfg.DB.NAME if self.cfg.DB.TESTNAME == '' else self.cfg.DB.TESTNAME
        val_dataloader = build_dataloader(self.cfg, 'test', shuffle=True, is_train=False, name=test)
        return [val_dataloader, ]

    def test_dataloader(self):
        test = self.cfg.DB.NAME if self.cfg.DB.TESTNAME == '' else self.cfg.DB.TESTNAME
        val_dataloader = build_dataloader(self.cfg, self.cfg.TEST.SET, is_train=False, name=test)
        return [val_dataloader, ]

    def get_jsTx(self, hA, hTx):
        hTjs = self.hand_wrapper.pose_to_transform(hA, False)
        N, num_j, _, _ = hTjs.size()
        jsTh = geom_utils.inverse_rt(mat=hTjs, return_mat=True)
        hTx_exp = geom_utils.se3_to_matrix(hTx).unsqueeze(1).repeat(1, num_j, 1, 1)
        jsTx = jsTh @ hTx_exp        
        return jsTx

    def sdf(self, hA, sdf_hA_jsTx, hTx):
        sdf = functools.partial(sdf_hA_jsTx, hA=hA, jsTx=self.get_jsTx(hA, hTx))
        return sdf
    
    def training_step(self, batch, batch_idx):
        losses, out = self.step(batch, batch_idx)
        losses = {'train_' + e: v for e,v in losses.items()}
        if self.trainer.is_global_zero:
            self.log_dict(losses)
            if self.global_step % self.hparams.TRAIN.PRINT_EVERY == 0:
                self.logger.print(self.global_step, self.current_epoch, losses, losses['train_loss'])
        return losses['train_loss']

    def test_step(self, *args):
        if len(args) == 3:
            batch, batch_idx, dataloader_idx = args
            if not isinstance(batch_idx, int):
                batch_idx = batch_idx[0]
                dataloader_idx = dataloader_idx[0]
        elif len(args) == 2:
            batch, batch_idx, = args
            if not isinstance(batch_idx, int):
                batch_idx = batch_idx[0]
            dataloader_idx = 0
        else:
            raise NotImplementedError

        image_name = '-'.join(batch['index'][0].split('/'))
        prefix = image_name
        losses, out = self.step(batch, 0)
        f_res = self.quant_step(out, batch)
        return f_res

    def test_epoch_end(self, outputs: List[Any], save_dir=None) -> None:
        save_dir = self.logger.local_dir if save_dir is None else save_dir
        arg_parser = default_argument_parser()
        arg_parser = slurm_utils.add_slurm_args(arg_parser)
        args = arg_parser.parse_args()
        suf = '_' + args.ckpt.split('/')[-1].split('_')[0]
        mean_list = mesh_utils.test_end_fscore(outputs, save_dir, suf)
        
    def validation_step(self, *args):
        return args

    def validation_step_end(self, batch_parts_outputs):
        args = batch_parts_outputs
        if len(args) == 3:
            batch, batch_idx, dataloader_idx = args
            if not isinstance(batch_idx, int):
                batch_idx = batch_idx[0]
                dataloader_idx = dataloader_idx[0]
        elif len(args) == 2:
            batch, batch_idx, = args
            if not isinstance(batch_idx, int):
                batch_idx = batch_idx[0]
            dataloader_idx = 0
        else:
            raise NotImplementedError
        prefix = '%d_%d' % (dataloader_idx, batch_idx)
        losses, out = self.step(batch, 0)
        losses = {'val_' + e: v for e,v in losses.items()}
        # val loss
        self.log_dict(losses, prog_bar=True, sync_dist=True)

        if self.trainer.is_global_zero:
            self.quant_step(out, batch)
            if batch_idx % 100 == 0:
                self.vis_step(out, batch, prefix)
        return losses

    def quant_step(self, out, batch, sdf=None):
        device = batch['cam_f'].device
        N = batch['cam_f'].size(0)
        hTx = get_hTx(self.cfg.MODEL.FRAME, batch)

        if sdf is None:
            camera = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=device)
            cTx = geom_utils.compose_se3(batch['cTh'], get_hTx(self.cfg.MODEL.FRAME, batch))
            contact_feat = batch['contact_label']

            zeros = torch.zeros([N, 3], device=device)
            hHand, hJoints = self.hand_wrapper(None, batch['hA'], zeros, mode='inner')
            xHand = mesh_utils.apply_transform(hHand, geom_utils.inverse_rt(hTx))

            # normal space, joint space jsTn, image space 
            sdf = functools.partial(self.dec, z=out['z'], hA=batch['hA'], jsTx=out['jsTx'], contact_feat=contact_feat, xHand=xHand, cTx=cTx, cam=camera)

        xObj = mesh_utils.batch_sdf_to_meshes(sdf, N)
        th_list = [.5/100, 1/100,]
        gt_pc = batch[self.obj_key][..., :3]

        hObj = mesh_utils.apply_transform(xObj, hTx) 
        hGt = mesh_utils.apply_transform(gt_pc, hTx)
        f_res = mesh_utils.fscore(hObj, hGt, num_samples=gt_pc.size(1), th=th_list)
        for th, th_f in zip(th_list, f_res[:-1]):
            self.log('f-%d' % (th*100), np.mean(th_f), sync_dist=True)
        self.log('cd', np.mean(f_res[-1]), sync_dist=True)
        return  [batch['indices'].tolist()] + f_res

    def vis_input(self, out, batch, prefix):
        N = len(batch['hObj'])
        device = batch['hObj'].device
        zeros = torch.zeros([N, 3], device=device)
        hHand, _ = self.hand_wrapper(None, batch['hA'], zeros, mode='inner')
        # self.logger.save_images(self.global_step, batch['image'], '%s_image' % prefix)
        # mesh_utils.dump_meshes(osp.join(self.logger.local_dir, '%d_%s/hand' % (self.global_step, prefix)), hHand)
        # mesh_utils.dump_meshes(osp.join(self.logger.local_dir, 'sdf_mesh/%s_hand' % (prefix)), hHand)
        # # self.logger.save_gif(self.global_step, image_list, '%s_inp' % prefix)
        return {'hHand': hHand}
    
    def vis_output(self, out, batch, prefix, cache={}):
        N = len(batch['hObj'])
        device = batch['hObj'].device
        zeros = torch.zeros([N, 3], device=device)
        hHand, hJoints = self.hand_wrapper(None, batch['hA'], zeros, mode='inner')
        cJoints = mesh_utils.apply_transform(hJoints, batch['cTh'])
        cache['hHand'] = hHand
        hTx = get_hTx(self.cfg.MODEL.FRAME, batch)  # 'norm'
        xHand = mesh_utils.apply_transform(hHand, geom_utils.inverse_rt(hTx))
        camera = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=device)
        cTx = geom_utils.compose_se3(batch['cTh'], get_hTx(self.cfg.MODEL.FRAME, batch))
        contact_feat = batch['contact_label']
        sdf = functools.partial(self.dec, z=out['z'], hA=batch['hA'], jsTx=out['jsTx'], contact_feat=contact_feat, xHand=xHand, cTx=cTx, cam=camera)
        xObj = mesh_utils.batch_sdf_to_meshes(sdf, N, bound=True)
        cache['xMesh'] = xObj
        hObj = mesh_utils.apply_transform(xObj, hTx)
        xHand.textures = mesh_utils.pad_texture(xHand, 'blue')
        xObj.textures = mesh_utils.pad_texture(xObj, 'yellow')
        xHoi = mesh_utils.join_scene([xHand, xObj])
        # mesh_utils.dump_meshes(osp.join(self.logger.local_dir, '%d_%s/obj' % (self.global_step, prefix)), hObj)
        # mesh_utils.dump_meshes(osp.join(self.logger.local_dir, 'sdf_mesh/%s_obj' % (prefix)), hObj)
        # image_list = mesh_utils.render_geom_rot(xHoi, scale_geom=True)
        # self.logger.save_gif(self.global_step, image_list, '%s_xHoi' % prefix)

        cHoi = mesh_utils.apply_transform(xHoi, cTx)
        image = mesh_utils.render_mesh(cHoi, camera)
        image_list = mesh_utils.render_geom_rot(cHoi, view_centric=True, cameras=camera)
        img_bg = torch.ones_like(batch['image'])
        # self.logger.save_images(self.global_step, image['image'], '%s_cam_mesh' % prefix, bg=batch['image'], mask=image['mask'])
        # self.logger.save_gif(self.global_step, image_list, 'gif/%s_cHoi' % prefix)
        # self.logger.save_images(self.global_step, image['image'], 'img/%s_cam_mesh_noimg' % prefix, bg=img_bg, mask=image['mask'],r=2.0)

        return cache

    def vis_step(self, out, batch, prefix):
        cache = self.vis_input(out, batch, prefix)
        cache = self.vis_output(out, batch, prefix, cache)
        return cache

    def step(self, batch, batch_idx):
        contact_feat = batch['contact_label']
        image_feat = self.enc(batch['image'], mask=batch['obj_mask'])
        xXyz = batch[self.sdf_key][..., :3]
        hTx = get_hTx(self.cfg.MODEL.FRAME, batch)
        N = len(batch['hObj'])
        device = batch['hObj'].device
        zeros = torch.zeros([N, 3], device=device)
        hHand, hJoints = self.hand_wrapper(None, batch['hA'], zeros, mode='inner')
        xHand = mesh_utils.apply_transform(hHand, geom_utils.inverse_rt(hTx))

        cTx = geom_utils.compose_se3(batch['cTh'], hTx)
        cameras = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=xXyz.device)
        hTjs = self.hand_wrapper.pose_to_transform(batch['hA'], False)
        N, num_j, _, _ = hTjs.size()

        jsTh = geom_utils.inverse_rt(mat=hTjs, return_mat=True)
        hTx = geom_utils.se3_to_matrix(hTx).unsqueeze(1).repeat(1, num_j, 1, 1)
        jsTx = jsTh @ hTx

        pred_sdf = self.dec(xXyz, image_feat, batch['hA'], contact_feat, xHand, cTx, cameras, jsTx=jsTx)
        ndcPoints = mesh_utils.transform_points(mesh_utils.apply_transform(xXyz, cTx), cameras)
        out = {self.sdf_key: pred_sdf, 'z': image_feat, 'jsTx': jsTx}

        loss, losses = 0., {}
        cfg = self.cfg.LOSS

        recon_loss = self.sdf_loss(pred_sdf, batch[self.sdf_key][..., -1:], ndcPoints,
            cfg.RECON, cfg.ENFORCE_MINMAX, )
        loss = loss + recon_loss
        losses['recon'] = recon_loss

        losses['loss'] = loss
        return losses, out

    def sdf_loss(self, sdf_pred, sdf_gt, ndcPoints, wgt=1, minmax=False, ):
        # recon loss
        mode = self.cfg.LOSS.OFFSCREEN  # [gt, out, idc] gt
        if mode == 'gt':
            pass
        elif mode == 'out':
            mask = torch.all(ndcPoints <= 1, dim=-1, keepdim=True) &\
                 torch.all(ndcPoints >= -1, dim=-1, keepdim=True)
            value = self.maxT if self.cfg.MODEL.OCC == 'sdf' else 1
            sdf_gt = mask * sdf_gt + (~mask) * value
        elif mode == 'idc':
            mask = torch.any(ndcPoints <= 1, dim=-1, keepdim=True) & \
                torch.any(ndcPoints >= -1, dim=-1, keepdim=True)
            sdf_pred = sdf_pred * mask  # the idc region to zero
            sdf_gt = sdf_gt * mask
        else:
            raise NotImplementedError

        if minmax or self.current_epoch >= self.cfg.TRAIN.EPOCH // 2:
            sdf_pred = torch.clamp(sdf_pred, self.minT, self.maxT)
            sdf_gt = torch.clamp(sdf_gt, self.minT, self.maxT)
        recon_loss = wgt * F.l1_loss(sdf_pred, sdf_gt)
        return recon_loss


def main(cfg, args):
    pl.seed_everything(cfg.SEED)
    
    model = CHOI(cfg)
    if args.ckpt is not None:
        print('load from', args.ckpt)
        model = model.load_from_checkpoint(args.ckpt, cfg=cfg, strict=False, map_location='cpu')

    # instantiate model
    if args.eval:
        logger = MyLogger(save_dir=cfg.OUTPUT_DIR,
                        name=os.path.dirname(cfg.MODEL_SIG),
                        version=os.path.basename(cfg.MODEL_SIG),
                        subfolder=cfg.TEST.DIR,
                        resume=True,
                        )
        trainer = pl.Trainer(gpus='0,',
                             default_root_dir=cfg.MODEL_PATH,
                             logger=logger,
                            #  resume_from_checkpoint=args.ckpt,
                             )
        print(cfg.MODEL_PATH, trainer.weights_save_path, args.ckpt)

        model.freeze()
        trainer.test(model=model, verbose=False)
    else:
        logger = MyLogger(save_dir=cfg.OUTPUT_DIR,
                        name=os.path.dirname(cfg.MODEL_SIG),
                        version=os.path.basename(cfg.MODEL_SIG),
                        subfolder=cfg.TEST.DIR,
                        resume=args.slurm or args.ckpt is not None,
                        )
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor='f-0',
            mode='max',
            save_last=True,
        )
        lr_monitor = LearningRateMonitor()

        max_epoch = cfg.TRAIN.EPOCH # max(cfg.TRAIN.EPOCH, cfg.TRAIN.ITERS // every_iter)
        trainer = pl.Trainer(
                             gpus=-1,
                             accelerator='dp',
                             # num_sanity_val_steps=0,
                             num_sanity_val_steps=2,
                             limit_val_batches=1.0,
                             # check_val_every_n_epoch=cfg.TRAIN.EVAL_EVERY,
                             check_val_every_n_epoch=1,
                             default_root_dir=cfg.MODEL_PATH,
                             logger=logger,
                             max_epochs=max_epoch,
                             callbacks=[checkpoint_callback, lr_monitor],
                             progress_bar_refresh_rate=0 if args.slurm else None,            
                             )
        trainer.fit(model)


if __name__ == '__main__':
    arg_parser = default_argument_parser()
    arg_parser = slurm_utils.add_slurm_args(arg_parser)
    args = arg_parser.parse_args()
    
    cfg = setup_cfg(args)
    save_dir = os.path.dirname(cfg.MODEL_PATH)
    slurm_utils.slurm_wrapper(args, save_dir, main, {'args': args, 'cfg': cfg}, resubmit=False)
