from rgvi import RGVI
from argparse import ArgumentParser
import os
import torch
import warnings
from PIL import Image
import cv2
import gc
from tqdm import tqdm
import torchvision as tv

warnings.filterwarnings('ignore')


parser = ArgumentParser()
parser.add_argument('--video', default=r'D:\SAM2-GUI-PySide6\workspace\20250709_140347_output_12 - Trim - Trim\frames', type=str, help='path to video frames')
parser.add_argument('--mask', default=r'D:\SAM2-GUI-PySide6\workspace\20250709_140347_output_12 - Trim - Trim\result_is_color_False_is_video_False_is_invert_False', type=str, help='path to mask frames')
parser.add_argument('--res', default='2K', choices=['240p', '480p', '2K'], help='input resolution')
parser.add_argument('--prompt', default=None, type=str, help='text prompt for generative model')
parser.add_argument('--output', default='./outputs/', type=str, help='path to output frames')
args = parser.parse_args()


def read_img(path):
    pic = Image.open(path).convert('RGB')
    transform = tv.transforms.ToTensor()
    return transform(pic)


def read_mask(path):
    pic = Image.open(path).convert('L')
    transform = tv.transforms.ToTensor()
    return transform(pic)

def is_any_mask(masks):
    """
    Check if any mask in the batch is non-zero.
    """
    return torch.any(masks > 0)

def inference(model, imgs, masks, res, prompt):
    with torch.no_grad():
        pred_imgs = model(imgs, masks, None, res, prompt)
    pred_imgs = [(pred_imgs[i].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype('uint8') for i in range(len(pred_imgs))]

    torch.cuda.empty_cache()
    gc.collect()
    return pred_imgs

def inference_slice(model, frame_ids, res, prompt, output_image_dir, output_video_dir,
                    slice_size=100, slice_overlap=10):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_video_dir, exist_ok=True)
    
    # call inference for each slice
    last_end = 0
    for start in range(0, len(frame_ids), slice_size - slice_overlap):
        # get range for current slice
        end = min(start + slice_size, len(frame_ids)) 
        start = max(0, end - slice_size)  # overlap with previous slice       

        print(f"Processing slice {start}: {end}")
        output_video_path = os.path.join(output_video_dir, f"{start}_{end}_{res}.mp4")
        if os.path.exists(output_video_path):
            print(f"Output video {output_video_path} already exists. Skipping inference.")
            continue

        # prepare data
        print(f"prepare data")
        imgs = []
        masks = []
        for index in range(start, end):
            mask_path = os.path.join(args.mask, '{:09d}.png'.format(frame_ids[index]))
            if index < last_end: # read output as padding
                img_path = os.path.join(output_image_dir, '{:09d}.png'.format(frame_ids[index]))
            else:
                img_path = os.path.join(args.video, '{:09d}.png'.format(frame_ids[index]))
            assert os.path.exists(img_path), f"Image {img_path} does not exist."
            assert os.path.exists(mask_path), f"Mask {mask_path} does not exist."

            imgs.append(read_img(img_path))
            masks.append(read_mask(mask_path))
        imgs = torch.stack(imgs)
        masks = torch.stack(masks)
        if not is_any_mask(masks):
            print(f"No masks found in frame clips start {start} to {end}. Skipping inference.")
            continue
        
        print(f"inference started for frames {start} to {end}")
        pred_imgs = inference(model, imgs, masks, res, prompt)

        # save output
        print(f"Saving output frames to {output_image_dir}")
        os.makedirs(output_image_dir, exist_ok=True)
        for i in range(end - start): # skip padding frames
            if(start + i < last_end): continue

            fpath = os.path.join(output_image_dir, f'{start + i:09d}.png')
            frame = cv2.cvtColor(pred_imgs[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(fpath, frame)

        # save to video
        print(f"Saving output video to {output_video_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = pred_imgs[0].shape[:2]
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
        for i in range(end - start):
            frame = cv2.cvtColor(pred_imgs[i], cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        video_writer.release()

    # merge all slices into a single video
    print(f"All slices processed. Merging into a single video...")
    output_video_merge_path = os.path.join(output_video_dir, f"{frame_ids[0]}_{frame_ids[-1]}_{res}_merge.mp4")
    output_video_path = os.path.join(output_video_dir, f"{frame_ids[0]}_{frame_ids[-1]}_{res}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = read_img(os.path.join(args.video, '{:09d}.png'.format(frame_ids[0]))).shape[1:3]
    video_merge_writer = cv2.VideoWriter(output_video_merge_path, fourcc, 30.0, (width, height * 2))
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
    for i in tqdm(range(len(frame_ids))):
        fpath = os.path.join(output_image_dir, '{:09d}.png'.format(frame_ids[i]))
        ori_path = os.path.join(args.video, '{:09d}.png'.format(frame_ids[i]))
        if not os.path.exists(fpath):
            fpath = ori_path
        assert os.path.exists(fpath), f"Original frame {fpath} does not exist."
        assert os.path.exists(ori_path), f"Original frame {ori_path} does not exist."
        
        
        frame = cv2.imread(fpath)
        ori_frame = cv2.imread(ori_path)
        frame = cv2.resize(frame, (width, height))
        ori_frame = cv2.resize(ori_frame, (width, height))

        # up down concat
        merged_frame = cv2.vconcat([ori_frame, frame])
        video_writer.write(frame)
        video_merge_writer.write(merged_frame)

    video_writer.release()
    video_merge_writer.release()
    print(f"Video saved to {output_video_path}")


if __name__ == '__main__':

    # set device
    torch.cuda.set_device(0)

    # define model
    model = RGVI().eval().cuda()

    # prepare data
    frame_ids = sorted([int(os.path.splitext(file)[0]) for file in os.listdir(args.video)])
    
    output_image_dir = os.path.join(args.output, 'images')
    output_video_dir = os.path.join(args.output, 'videos')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_video_dir, exist_ok=True)
    # inference(model, frame_ids, args.res, args.prompt, output_image_dir, output_video_dir)
    inference_slice(model, frame_ids, args.res, args.prompt, output_image_dir, output_video_dir)

    

    

