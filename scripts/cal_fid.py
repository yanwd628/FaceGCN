import os, sys
import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from basicsr.data import build_dataset
from basicsr.metrics.fid import calculate_fid, extract_inception_features, load_patched_inception_v3


def calculate_fid_folder(folders):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()

    parser.add_argument('--fid_stats', type=str, default="/root/yanwd_data/projects/FaceGCN/experiments/pretrained_models/inception_FFHQ_512-f7b384ab.pth",help='Path to the dataset fid statistics.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_sample', type=int, default=50000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--backend', type=str, default='disk', help='io backend for dataset. Option: disk, lmdb')
    args = parser.parse_args()

    # inception model
    inception = load_patched_inception_v3(device)

    for folder in folders:
        # create dataset
        print(f"Testing {folder}")
        opt = {}
        opt['name'] = 'SingleImageDataset'
        opt['type'] = 'SingleImageDataset'
        opt['dataroot_lq'] = folder
        opt['io_backend'] = dict(type=args.backend)
        opt['mean'] = [0.5, 0.5, 0.5]
        opt['std'] = [0.5, 0.5, 0.5]
        dataset = build_dataset(opt)

        # create dataloader
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            sampler=None,
            drop_last=False)
        args.num_sample = min(args.num_sample, len(dataset))
        total_batch = math.ceil(args.num_sample / args.batch_size)

        def data_generator(data_loader, total_batch):
            for idx, data in enumerate(data_loader):
                if idx >= total_batch:
                    break
                else:
                    yield data['lq']

        features = extract_inception_features(data_generator(data_loader, total_batch), inception, total_batch, device)
        features = features.numpy()
        total_len = features.shape[0]
        features = features[:args.num_sample]
        # print(f'Extracted {total_len} features, ' f'use the first {features.shape[0]} features to calculate stats.')

        sample_mean = np.mean(features, 0)
        sample_cov = np.cov(features, rowvar=False)

        # load the dataset stats
        stats = torch.load(args.fid_stats)
        real_mean = stats['mean']
        real_cov = stats['cov']

        # calculate FID metric
        fid = calculate_fid(sample_mean, sample_cov, real_mean, real_cov)
        print('fid:', fid)

if __name__ == '__main__':

    exp_name = "test_facegcn_1025_20k"
    folders = [
               f'/root/yanwd_data/projects/FaceGCN/results/{exp_name}/visualization/LFW-Test/',
               f'/root/yanwd_data/projects/FaceGCN/results/{exp_name}/visualization/CelebChild/',
               f'/root/yanwd_data/projects/FaceGCN/results/{exp_name}/visualization/WebPhoto/',
               ]

    # change folders to your paths

    calculate_fid_folder(folders)
