import torch
from diffusers.utils import load_image
from diffusers import AutoPipelineForInpainting
import PIL
from PIL import Image
import os
import numpy as np
import re
from tqdm import tqdm
import json
import cv2

def concatenate_images_horizontally(image1, image2):
    new_image = Image.new('RGB', (2 * image1.size[0],  image1.size[1]))

    # 将图像1复制到新图像中的左侧
    new_image.paste(image1, (0, 0))

    # 将图像2复制到新图像中的右侧
    new_image.paste(image2, (image1.size[0], 0))
    return new_image

def text_under_image(image: np.ndarray, text: str, text_color = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .1)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_COMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h,:w] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    if w > textsize[0]:
        text_x = (w - textsize[0]) // 2
    else:
        text_x = 0
    cv2.putText(img, text, (text_x, text_y ), font, 0.5, text_color, 1)
    return Image.fromarray(img)



pipeline = AutoPipelineForInpainting.from_pretrained(
    "models/sd1.4", safety_checker=None, torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

dir_name = '/data/lzy/coco/'
img_folder_path = os.path.join(dir_name,'people_images_1')
mask_folder_path = os.path.join(dir_name,'masks_1')
bad_name_folder_path = os.path.join(dir_name,'bad_people_images_2')
comparison_images_folder_path = os.path.join(dir_name,'comparison_2')
img_name_list = sorted(os.listdir(img_folder_path))
mask_name_list = sorted(os.listdir(mask_folder_path))

annotation_path = '/data/lzy/coco/annotations/people_captions_train2014.json'
with open('/data/lzy/coco/annotations/people_captions_train2014_1.json', 'r') as file:
    data = json.load(file)
num = 0
for img_name, mask_name in tqdm(zip(img_name_list,mask_name_list)):
    if num % 100 == 0:
        current_comparison_images_folder = os.path.join(comparison_images_folder_path, f"{num//100:04d}")
        os.makedirs(current_comparison_images_folder,exist_ok=True)
    
    pattern = r"_(\d+)\."      
    match = re.search(pattern, img_name)
 
    image_id = int(match.group(1))
    img_path = os.path.join(img_folder_path,img_name)
    mask_path = os.path.join(mask_folder_path,mask_name)
    # img_path = '/data/lzy/coco/people_images_1/COCO_train2014_000000001732.jpg'
    # mask_path = '/data/lzy/coco/masks_1/COCO_train2014_000000001732.jpg'
    init_image = Image.open(img_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("RGB")

    init_image = init_image.resize((512,512), resample=Image.BICUBIC, reducing_gap=1)
    # init_image.save(f'/data/lzy/coco/resize_1/COCO_train2014_{image_id:012d}.jpg')
    # init_image.save(f'./working_dir/1127/COCO_train2014_{image_id:012d}.jpg')
    # init_image.save(f'./1120/COCO_train2014_{image_id:012d}.jpg')
    mask_image = mask_image.resize((512,512), resample=PIL.Image.LANCZOS)
    init_image_arr = np.array(init_image)

    # Convert mask to grayscale NumPy array
    mask_image_arr = np.array(mask_image.convert("L"))
    # Add a channel dimension to the end of the grayscale mask
    mask_image_arr = mask_image_arr[:, :, None]
    # Binarize the mask: 1s correspond to the pixels which are repainted
    mask_image_arr = mask_image_arr.astype(np.float32) / 255.0
    mask_image_arr[mask_image_arr < 0.5] = 0
    mask_image_arr[mask_image_arr >= 0.5] = 1

    generator = torch.Generator("cuda").manual_seed(42)
    caption = [item for item in data if item['image_id'] == image_id][0]['caption']
    for i in range(10):
        image = pipeline(prompt="a face", image=init_image, mask_image=mask_image, strength = 0.1,generator=generator).images[0]
        # image.save(f'./working_dir/1127/COCO_train2014_{image_id:012d}_repaint.jpg')
        # image.save(f'./1120/COCO_train2014_{image_id:012d}_{i}.jpg')
        image.save(os.path.join(bad_name_folder_path, f'COCO_train2014_{image_id:012d}_{i}.jpg'))
        concat_image = concatenate_images_horizontally(init_image,image)
        concat_image = text_under_image(np.array(concat_image),caption)
        concat_image.save(os.path.join(current_comparison_images_folder,f'COCO_train2014_{image_id:012d}_{i}.jpg'))
        num = num + 1