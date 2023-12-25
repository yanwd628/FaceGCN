import cv2
import glob
from tqdm import tqdm
from basicsr.metrics import calculate_niqe

img_path = r"/root/yanwd_data/projects/FaceGCN/results/*.png" # change to your path

img_names = glob.glob(img_path)
img_names.sort()
crop_border = 0

mean_niqe = 0.
for i in tqdm(range(len(img_names))):
    img = cv2.imread(img_names[i])
    mean_niqe += calculate_niqe(img, crop_border=crop_border)

mean_niqe /= float(len(img_names))

print(mean_niqe)