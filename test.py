import os
import shutil
import cv2
import re
import json
import pandas as pd 
from torchvision import transforms
import torch
from tqdm import tqdm
from batch_face import RetinaFace
from PIL import Image, ImageDraw
import torch.nn.functional as F
# def numerical_sort(string):
#     # 使用正则表达式提取文件名中的数字部分
#     match = re.search(r'(\d+)\.jpg$', string)
#     if match:
#         return int(match.group(1))  # 返回提取的数字部分作为整数
#     else:
#         return 0  
# def organize_images(source_folder, destination_folder, images_per_folder=1000):
#     # 创建目标文件夹
#     os.makedirs(destination_folder, exist_ok=True)

#     # 遍历源文件夹中的图片文件
#     counter = 0
#     image_files = sorted([file for file in os.listdir(source_folder) if file.endswith('.jpg')],key=numerical_sort)
#     for index, image_file in enumerate(image_files):
#         # 计算目标子文件夹的索引
#         # 将图片移动到目标子文件夹中
#         source_path = os.path.join(source_folder, image_file)
#         tmp_destination_path = os.path.join(destination_folder, image_file)
#         shutil.move(source_path, tmp_destination_path)
#         counter +=1
#         print(f"Moved {image_file} to {destination_folder}")
#         if counter == images_per_folder:
#             return False
#     else:
#         return True

# # 调用函数进行图片组织
# folder_counter = 0
# finish = False
# source_folder = "/data/lzy/improved_aesthetics_6.5plus/regularization_images"

# while not finish:
#     destination_folder = f"/data/lzy/improved_aesthetics_6.5plus/{folder_counter:04d}"
#     # destination_folder = f"tmp/{folder_counter:04d}"
#     finish = organize_images(source_folder, destination_folder)
#     folder_counter += 1
#     print(folder_counter)


# for i in range(1,10):
#     dir_name = f'/data/lzy/LAION-FACE-100k/split_00000/0000{i}'
#     print(dir_name)
#     # dir_name = 'tmp'
#     # os.makedirs(os.path.join(dir_name,'images'))
#     # os.makedirs(os.path.join(dir_name,'jsons'))
#     # os.makedirs(os.path.join(dir_name,'prompts'))
#     data_name = os.listdir(dir_name)
#     for data in data_name:
#         if 'jpg' in data:
#             shutil.move(os.path.join(dir_name,data),os.path.join(dir_name,'images'))
#         elif 'txt' in data:
#             shutil.move(os.path.join(dir_name,data),os.path.join(dir_name,'prompts'))
#         elif 'json' in data and 'stat' not in data:
#             shutil.move(os.path.join(dir_name,data),os.path.join(dir_name,'jsons'))

# for i in range(1,10):
#     folder_path = f'/data/lzy/LAION-FACE-100k/split_00000/0000{i}/masks'  # 替换为你的文件夹路径
#     # folder_path = 'tmp'
#     # 遍历文件夹中的文件
#     for filename in os.listdir(folder_path):
#         if filename.endswith("_mask.jpg"):
#             # 构造新的文件名
#             new_filename = filename.replace("_mask", "")
            
#             # 构造旧文件和新文件的完整路径
#             old_file_path = os.path.join(folder_path, filename)
#             new_file_path = os.path.join(folder_path, new_filename)
            
#             # 重命名文件
#             os.rename(old_file_path, new_file_path)
# regularization_file = '/data/lzy/improved_aesthetics_6.5plus/regularization_images.jsonl'
# filename = os.path.basename(regularization_file)
# directory = os.path.dirname(regularization_file)

# with open(regularization_file, 'r') as f:
#     for key, row in enumerate(f):
#         data = json.loads(row)
#         print(key)
#         sub_folder = key//1000
#         original_image_path = data['file_name']
#         filename = os.path.basename(original_image_path)
#         directory = os.path.dirname(original_image_path)
#         modified_image_path = os.path.join(directory,f"{sub_folder:04d}",filename)
#         print(modified_image_path)
#         break
# from PIL import Image

# # 创建白色图片
# image = Image.new('RGB', (512, 512), (255, 255, 255))

# # 保存图片
# image.save('/data/lzy/improved_aesthetics_6.5plus/mask_image.jpg')
# folder_path = f'/data/lzy/LAION-FACE-100k/split_00000/00000' 
# for i in range(10):
#     folder_path = f'/data/lzy/LAION-FACE-100k/split_00000/0000{i}' 
#     bad_path = os.path.join(folder_path, 'bad_images')
#     print(len(os.listdir(bad_path)))


# data = data.drop_duplicates(subset=['file_name', 'caption'], keep='first')
# print(data)
# data.to_json('/data/lzy/improved_aesthetics_6.5plus/regularization_images1.jsonl', orient='records', lines=True)

# duplicated_row = data[data.duplicated(['file_name','caption'],keep=False)]
# print(duplicated_row)
# filtered_rows = data[data['file_name'] == '/data/lzy/improved_aesthetics_6.5plus/regularization_images/0.jpg']
# print(filtered_rows)
# print(filtered_rows.iloc[0]["caption"] == filtered_rows.iloc[1]["caption"])
# deduplicated_df = data.drop_duplicates(subset=['file_name', 'caption'], keep='first')
# print(data)


# dir_name = '/data/lzy/improved_aesthetics_6.5plus/0008'
# img_list = sorted(os.listdir(dir_name))
# for img_name in img_list:
#     num = int(img_name[0:-4])
#     if num < 8000:
#         os.remove(os.path.join(dir_name,img_name))

# data = pd.read_json('/data/lzy/improved_aesthetics_6.5plus/regularization_images1.jsonl', lines=True) 
# for i in range(10000):
#     filtered_rows = data[data['file_name'] == f'/data/lzy/improved_aesthetics_6.5plus/regularization_images/{i}.jpg']
#     if filtered_rows.shape[0]!=1:
#         print(i)
#         print(filtered_rows)
#         break
# mask_transforms = transforms.Compose(
#     [
#         transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
#         transforms.RandomCrop(512),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ]
# )
# file_name = '/data/lzy/LAION-FACE-100k/split_00000/00000/masks/000000096.jpg'
# mask = Image.open(file_name).convert('RGB')
# mask = mask_transforms(mask)
# for dim1 in mask:
#     for dim2 in dim1:
#         for element in dim2:
#             if element == 1:
#                 print(element)
# print(torch.count_nonzero(mask == 255))
# mask = mask/255
# print(torch.count_nonzero(mask == 1))

# data = pd.read_json('/data/lzy/improved_aesthetics_6.5plus1/regularization_images.jsonl', lines=True) 
# data = data.head(77760)
# data.to_json('/data/lzy/improved_aesthetics_6.5plus1/regularization_images2.jsonl', orient='records', lines=True)

# loss = F.mse_loss(batch["weight"].view(-1,1,1,1) * mask * model_pred.float(), batch["weight"].view(-1,1,1,1) * mask * target.float(), reduction="mean")
# positive_index = []
# negative_index = []
# for i,weight in enumerate(batch["weight"]):
#     if weight == 1:
#         positive_index.append(i)
#     else:
#         negative_index.append(i)
# if len(positive_index) != 0:
#     loss_positive = F.mse_loss((batch["weight"].view(-1,1,1,1) * mask * model_pred.float())[positive_index], (batch["weight"].view(-1,1,1,1) * mask * target.float())[positive_index], reduction="mean")
# else:
#     loss_positive = torch.tensor(0)
# if len(negative_index) != 0:
#     loss_negative = F.mse_loss((batch["weight"].view(-1,1,1,1) * mask * model_pred.float())[negative_index], (batch["weight"].view(-1,1,1,1) * mask * target.float())[negative_index], reduction="mean")
# else:
#     loss_negative = torch.tensor(0)

# mask = Image.open('/data/lzy/LAION-FACE-100k/split_00000/00000/masks/000000096.jpg').convert('RGB')
# mask_transforms = transforms.Compose(
#     [
#         transforms.Resize(512, interpolation=transforms.InterpolationMode.NEAREST),
#         transforms.CenterCrop(512),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ]
# )
# mask = mask_transforms(mask)
# face_mask_values = torch.stack([mask])
# face_mask_values = face_mask_values.to(memory_format=torch.contiguous_format).float()
# print(mask.dtype)
# mask = face_mask_values[:,0:1]
# # for dim in mask[0][0]:
# #     for element in dim:
# #         if element!=0:
# #             print(element)
# mask = F.interpolate(mask, size=(64,64), mode='nearest')
# print(torch.sum(mask))

# words = ["man","woman","boy","girl"]
# counter = 0
# people_annotation_list = []
# for i in range(len(json_data["annotations"])):
#     if any(word in json_data["annotations"][i]["caption"] for word in words):
#         people_annotation_list.append(json_data["annotations"][i])
#         counter += 1
#         if counter == 200:
#             break

# json_path = '/data/myb/coco/annotations/captions_train2014.json'
# with open(json_path, "r") as f1:
#     json_str = f1.read()
# json_data = json.loads(json_str)

# words = ["man","woman","boy","girl"]
# image_id_set = set()
# people_annotation_list = []
# for i in range(len(json_data["annotations"])):
#     if any(word in json_data["annotations"][i]["caption"] for word in words):
#         if json_data["annotations"][i]["image_id"] in image_id_set:
#             continue
#         people_annotation_list.append(json_data["annotations"][i])
#         image_id_set.add(json_data["annotations"][i]["image_id"])
#         if len(image_id_set) == 500:
#             break
# people_json_path = '/data/lzy/coco/annotations/people_captions_train2014_1.json'
# people_json_data = json.dumps(people_annotation_list)
# with open(people_json_path, "w") as json_file:
#     json_file.write(people_json_data)

# for i in range(len(people_annotation_list)):
#     image_id = people_annotation_list[i]["image_id"]
#     src_file_name = f'/data/myb/coco/train2014/COCO_train2014_{image_id:012d}.jpg'
#     des_file_name = f'/data/lzy/coco/people_images_1/COCO_train2014_{image_id:012d}.jpg'
#     shutil.copy(src_file_name,des_file_name)



# json_path = '/data/myb/coco/annotations/captions_train2014.json'
# json_path_people = '/data/lzy/coco/annotations/people_captions_train2014.json'
# with open(json_path, "r") as f1:
#     json_str = f1.read()
# json_data = json.loads(json_str)

# with open(json_path_people, "r") as f2:
#     json_str = f2.read()
# json_people_annotation = json.loads(json_str)

# for i in range(len(json_people_annotation)):
#     image_id = json_people_annotation[i]["image_id"]
#     src_file_name = f'/data/myb/coco/train2014/COCO_train2014_{image_id:012d}.jpg'
#     des_file_name = f'/data/lzy/coco/people_images/COCO_train2014_{image_id:012d}.jpg'
#     shutil.copy(src_file_name,des_file_name)




# data = pd.read_json('/data/myb/coco/annotations/captions_train2014.json', lines=True) 
# data = data.head(10)
# print(data)

# file_path = '/data/lzy/LAION-FACE-100k/split_00000/00000/comparison'
# file_name_list = sorted(os.listdir(file_path))
# for i in tqdm(range(len(file_name_list))):
#     current_path = f'/data/lzy/LAION-FACE-100k/split_00000/00000/comparisons/{i//100}'
#     if i%100 ==0:
#         os.makedirs(current_path,exist_ok=True)
#     src_path = os.path.join(file_path,file_name_list[i])
#     tgt_path = os.path.join(current_path,file_name_list[i])
#     shutil.copy(src_path,tgt_path)

# annotation_path = '/data/lzy/coco/annotations/people_captions_train2014.json'
# with open(annotation_path) as f:
#     image_info = json.load(f)
#     prompt = image_info["caption"]
#     if prompt is None:
#         prompt = ""

# json_path = '/data/myb/coco/annotations/captions_train2014.json'
# with open(json_path, "r") as f1:
#     json_str = f1.read()
# json_data = json.loads(json_str)
# detector = RetinaFace(gpu_id=0)
# words = ["man","woman","boy","girl","people","men","women","boys","girls"]
# # pattern = r"\b(" + "|".join(map(re.escape, words)) + r")\b"
# pattern = r"(?i)(?:(?:^|\s)(" + "|".join(map(re.escape, words)) + r")(?:\s|$))"
# image_id_set = set()
# people_annotation_list = []
# for i in range(len(json_data["annotations"])):
#     if re.match(pattern, json_data["annotations"][i]["caption"]):
#         if json_data["annotations"][i]["image_id"] in image_id_set:
#             continue
#         src_file_name = f'/data/myb/coco/train2014/COCO_train2014_{json_data["annotations"][i]["image_id"]:012d}.jpg'
#         img = cv2.imread(src_file_name)
#         mask_image = Image.new('L', (img.shape[1],img.shape[0]), 0)
#         faces = detector(img, cv=False) # set cv to False for rgb input, the default value of cv is False
#         if not faces:
#             continue
#         # print(i)
#         # print(json_data["annotations"][i]["caption"])
#         box, landmarks, score = faces[0]
#         draw = ImageDraw.Draw(mask_image)
#         draw.rectangle(box, fill=255)
#         mask_image.save(f'/data/lzy/coco/masks_1/COCO_train2014_{json_data["annotations"][i]["image_id"]:012d}.jpg')
#         people_annotation_list.append(json_data["annotations"][i])
#         if len(people_annotation_list)%1000 == 0:
#             print(len(people_annotation_list))
#         image_id_set.add(json_data["annotations"][i]["image_id"])
#         des_file_name = f'/data/lzy/coco/people_images_1/COCO_train2014_{json_data["annotations"][i]["image_id"]:012d}.jpg'
#         shutil.copy(src_file_name,des_file_name)
#         # if len(image_id_set) == 500:
#         #     break
# people_json_path = '/data/lzy/coco/annotations/people_captions_train2014_1.json'
# people_json_data = json.dumps(people_annotation_list)
# with open(people_json_path, "w") as json_file:
#     json_file.write(people_json_data)
# print(len(people_annotation_list))

# with open('/data/lzy/coco/annotations/people_captions_train2014.json', 'r') as file:
#     data = json.load(file)
# # words = [" man "," woman "," boy "," girl "]

# desired_image_id = 529
# caption = [item for item in data if item['image_id'] == desired_image_id][0]['caption']
# print(caption)


# parent_folder = '/data/lzy/coco/bad_people_images_1/'

# # 获取父文件夹下的所有子文件夹
# subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

# # 遍历子文件夹，将每个子文件夹中的文件移动到父文件夹下
# for subfolder in subfolders:
#     # 获取子文件夹中的所有文件
#     files = [f.path for f in os.scandir(subfolder) if f.is_file()]
    
#     # 移动文件到父文件夹下
#     for file in files:
#         shutil.move(file, parent_folder)
        
#     # 删除空的子文件夹
#     shutil.rmtree(subfolder)

# 11.27
# data_path = '/data/ksq/celeba/celeba/img_align_celeba/img_align_celeba/'
# image_list = os.listdir(data_path)
# print(len(image_list))
# img_path = os.path.join(data_path,image_list[1])
# image = Image.open(img_path)
# image.save(os.path.join('./working_dir/1127',image_list[1]))
detector = RetinaFace(gpu_id=0)
file_name = 'working_dir/1127/037886.jpg'
img = cv2.imread(file_name)
mask_image = Image.new('L', (img.shape[1],img.shape[0]), 0)
faces = detector(img, cv=False) # set cv to False for rgb input, the default value of cv is False
box, landmarks, score = faces[0]
draw = ImageDraw.Draw(mask_image)
draw.rectangle(box, fill=255)
mask_image.save(f'working_dir/1127/037886_mask.jpg')