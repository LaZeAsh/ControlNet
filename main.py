from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
from controlnet_aux import PidiNetDetector, HEDdetector
from diffusers.utils import load_image
from huggingface_hub import HfApi
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import cv2
import os
import random


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


controlnet_conditioning_scale = 1.0  
prompt = "Jerry jumping off a table"
negative_prompt = 'Tom'


eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")


controlnet = ControlNetModel.from_pretrained(
    "xinsir/controlnet-scribble-sdxl-1.0",
    torch_dtype=torch.float16
)

# when test with other base model, you need to change the vae also.
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    safety_checker=None,
    torch_dtype=torch.float16,
    scheduler=eulera_scheduler,
)

# you can use either hed to generate a fake scribble given an image or a sketch image totally draw by yourself

if random.random() > 0.5:
  # Method 1 
  # if you use hed, you should provide an image, the image can be real or anime, you extract its hed lines and use it as the scribbles
  # The detail about hed detect you can refer to https://github.com/lllyasviel/ControlNet/blob/main/gradio_fake_scribble2image.py
  # Below is a example using diffusers HED detector

  image_path = Image.open("cat.jpeg")
  processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
  controlnet_img = processor(image_path, scribble=False)
  controlnet_img.save("test.png")

  # following is some processing to simulate human sketch draw, different threshold can generate different width of lines
  controlnet_img = np.array(controlnet_img)
  controlnet_img = nms(controlnet_img, 127, 3)
  controlnet_img = cv2.GaussianBlur(controlnet_img, (0, 0), 3)

  # higher threshold, thiner line
  random_val = int(round(random.uniform(0.01, 0.10), 2) * 255)
  controlnet_img[controlnet_img > random_val] = 255
  controlnet_img[controlnet_img < 255] = 0
  controlnet_img = Image.fromarray(controlnet_img)

else:
  # Method 2
  # if you use a sketch image total draw by yourself
  control_path = "handdraw.png"
  controlnet_img = Image.open(control_path) # Note that the image must be black-white(0 or 255), like the examples we list

# must resize to 1024*1024 or same resolution bucket to get the best performance
width, height  = controlnet_img.size
ratio = np.sqrt(1024. * 1024. / (width * height))
new_width, new_height = int(width * ratio), int(height * ratio)
controlnet_img = controlnet_img.resize((new_width, new_height))

images = pipe(
    prompt,
    negative_prompt=negative_prompt,
    image=controlnet_img,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    width=new_width,
    height=new_height,
    num_inference_steps=30,
    ).images

images[0].save(f"final.png")
