import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets_inr
from utils import flow_viz
from utils import frame_utils
from torchvision.utils import save_image

from sepflow import SepFlow
from utils.utils import InputPadder, forward_interpolate, compute_out_of_boundary_mask

@torch.no_grad()
def create_sintel_submission(model, iters=32, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()


            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)



@torch.no_grad()
def validate_sintel_inr(model, image_root, flow_root,occlu_root=None, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets_inr.MpiSintel(split='training', dstype=dstype, image_root=image_root, flow_root=flow_root,occlu_root=occlu_root)
        epe_list = []
        if occlu_root:
            matched_epe_list = []
            unmatched_epe_list = []

        for val_id in tqdm(range(len(val_dataset)),desc=f'Validating Sintel {dstype}'):
            if occlu_root:
                image1, image2, flow_gt, valid, noc_valid = val_dataset[val_id]

                # compuate in-image-plane valid mask
                in_image_valid = compute_out_of_boundary_mask(flow_gt.unsqueeze(0)).squeeze(0)  # [H, W]
            else:
                image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters)
            flow = padder.unpad(flow_pr[0]).cpu()
            # save flow_pre
            flow_pre_png = flow_viz.flow_to_image(flow.permute(1, 2, 0).numpy())
            flow_gt_png = flow_viz.flow_to_image(flow_gt.permute(1, 2, 0).numpy())
            # save_image(torch.from_numpy(flow_pre_png).permute(2, 0, 1)/255, f'logs/test/{dstype}/flow_pre_{val_id}.png')
            save_image([torch.from_numpy(flow_pre_png).permute(2, 0, 1)/255,torch.from_numpy(flow_gt_png).permute(2, 0, 1)/255], f'logs/test/{dstype}/flow_res_{val_id}.png')

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())
            if occlu_root:
                matched_valid_mask = (noc_valid > 0.5) & (in_image_valid > 0.5)

                if matched_valid_mask.max() > 0:
                    matched_epe_list.append(epe[matched_valid_mask].cpu().numpy())
                    unmatched_epe_list.append(epe[~matched_valid_mask].cpu().numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        # px1 = np.mean(epe_all<1)
        # px3 = np.mean(epe_all<3)
        # px5 = np.mean(epe_all<5)

        # print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype+'_all'] = np.mean(epe_list)

        if occlu_root:
            matched_epe = np.mean(np.concatenate(matched_epe_list))
            unmatched_epe = np.mean(np.concatenate(unmatched_epe_list))
            results[dstype + '_matched'] = matched_epe
            results[dstype + '_unmatched'] = unmatched_epe
            print('===> Validatation Sintel (%s): all epe: %.3f matched epe: %.3f, unmatched epe: %.3f' % (
                dstype, epe, matched_epe, unmatched_epe))
        else:
            print('===> Validatation Sintel (%s): all epe: %.3f' % (dstype, epe))


    with open('logs/test/sintel_inr.txt', 'w') as f:
        f.write(str(results))

    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--image_root', help="images in dataset")
    parser.add_argument('--flow_root', help="flows in dataset")
    parser.add_argument('--occlu_root', default=None, help="occlusion maps in dataset")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(SepFlow(args))
    checkpoint = torch.load(args.model)
    msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(msg)

    print(args)

    # create output directory
    os.makedirs('results/test/clean', exist_ok=True)
    os.makedirs('results/test/final', exist_ok=True)

    model.cuda()
    model.eval()

    with torch.no_grad():
        if args.dataset == 'sintel':
            create_sintel_submission(model.module)



    with torch.no_grad():
        if args.dataset == 'sintel_inr':
            validate_sintel_inr(model.module, image_root=args.image_root, flow_root=args.flow_root, occlu_root=args.occlu_root)



