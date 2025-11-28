import argparse
import os
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline


def to_pil(tensor):
    # tensor: (3,H,W) in [-1,1] or uint8 PIL
    if hasattr(tensor, 'convert'):
        return tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='path to trained model directory')
    parser.add_argument('--input_image', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--out', type=str, default='out.png')
    parser.add_argument('--num_inference_steps', type=int, default=20)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model_dir, torch_dtype=torch.float16 if device=='cuda' else torch.float32)
    pipe = pipe.to(device)

    img = Image.open(args.input_image).convert('RGB')
    out = pipe(args.prompt, image=img, num_inference_steps=args.num_inference_steps, guidance_scale=7.0).images[0]
    out.save(args.out)
    print('Saved', args.out)
