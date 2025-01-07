from PIL import Image
import os


def convert_png_to_jpg(input_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png'):
            # 构建输入文件的路径
            input_path = os.path.join(input_folder, file_name)

            # 构建输出文件的路径，并将后缀改为.jpg
            output_name = os.path.splitext(file_name)[0] + '.jpg'
            output_path = os.path.join(output_folder, output_name)

            # 打开并转换图片格式
            with Image.open(input_path) as img:
                img = img.convert('RGB')  # 将图片转换为RGB格式
                img.save(output_path, 'JPEG')  # 保存为JPEG格式


# 指定输入和输出文件夹路径
input_folder = '/media/g/mydata/branch_ly/JPEGImages'
output_folder = '/media/g/mydata/branch_ly/JPEGImages_jpg'

# 调用函数进行转换
convert_png_to_jpg(input_folder, output_folder)
