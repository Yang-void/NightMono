import os
from PIL import Image
from torchvision import transforms
import torch


class Resize:
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.resize = transforms.Resize(size=size, interpolation=interpolation)

    def __call__(self, image):
        return self.resize(image)


class Crop:
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, image):
        return image.crop((self.left, self.top, self.left + self.width, self.top + self.height))


def preprocess_image(image_path, crop_height, crop_width, resize_height, resize_width):
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    
    # 创建裁剪操作
    crop = Crop(top=0, left=0, height=crop_height, width=crop_width)
    
    # 创建resize操作
    resize = Resize(size=(resize_height, resize_width))
    
    # 先裁剪，再resize
    cropped_image = crop(image)
    resized_image = resize(cropped_image)
    
    # 转换为Tensor并返回
    transform_to_tensor = transforms.ToTensor()
    processed_image_tensor = transform_to_tensor(resized_image)
    
    return processed_image_tensor, resized_image


def process_images_in_folder(input_folder, output_folder, crop_height, crop_width, resize_height, resize_width):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有图像文件
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        
        # 确保只处理图像文件
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {image_path}...")
            
            # 预处理图像
            processed_image_tensor, processed_image = preprocess_image(image_path, crop_height, crop_width, resize_height, resize_width)
            
            # 保存处理后的图像
            output_path = os.path.join(output_folder, filename)
            processed_image.save(output_path)  # 保存为图像文件
            
            # 如果你需要处理的图像会输入到模型中，可以在这里调用模型进行推理
            # 例如：model(processed_image_tensor.unsqueeze(0))  # 添加批次维度


# 配置裁剪和resize的大小
crop_height = 768
crop_width = 1280
resize_height = 320
resize_width = 544

# 输入和输出文件夹路径
input_folder = "/root/revert/night_new"
output_folder = "/root/revert/night_new_crop"

# 处理文件夹中的所有图像
process_images_in_folder(input_folder, output_folder, crop_height, crop_width, resize_height, resize_width)
