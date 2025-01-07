from PIL import Image
import numpy as np

# 读取图片
# image_path = '/home/g/aachen_000000_000019_gtFine_labelTrainIds.png'  # 示例0-255 2048*1024
image_path = '/media/g/mydata/branch_ly/SegmentationClassPNG/000001.png'  # 0和1 1280*720
# image_path = '/media/g/mydata/results/ddr_fordata_06g19/alpha0.3/test_000001.png'  # 3通道
# image_path = '/home/g/segmentation/mmsegmentation/output.png'  # 3通道
image = Image.open(image_path)

# 将数组中的0和1对调
image_array = np.array(image)
inverted_array = 1 - image_array
# inverted_image = Image.fromarray(inverted_array)
inverted_image = Image.fromarray(inverted_array.astype(np.uint8) * 255)
inverted_image.save('/media/g/mydata/inverted_out.png')

# 0和1 转换为 0和255
# new_image_array = image_array * 255
# new_image = Image.fromarray(new_image_array)
# new_image = Image.fromarray(inverted_array.astype(np.uint8) * 255)
# new_image.save('/media/g/mydata/out.png')

# 打印数组
print(image_array)
