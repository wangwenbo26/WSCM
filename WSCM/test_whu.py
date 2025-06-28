import os
import torch
from util.encoder import Encoder
from util.decoder import Decoder
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
from util.CFDM import CFDM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dir_vi_test = "datasets/whu/vi_test"
dir_ir_test = "datasets/whu/ir_test"
fus_directory = 'fusion_image/whu'
os.makedirs(fus_directory, exist_ok=True)

encoder = Encoder()
decoder = Decoder()
cfmd = CFDM(in_channels=512, head_dim=512, num_heads=8)

encoder.load_state_dict(torch.load('model_save/whu/encoder.pth'))
decoder.load_state_dict(torch.load('model_save/whu/decoder.pth'))
cfmd.load_state_dict(torch.load('model_save/whu/CFMD.pth'))

encoder.to(device)
decoder.to(device)
cfmd.to(device)

transform = transforms.Compose([
transforms.Resize((512, 512)),
transforms.ToTensor()
])

def fusion_images(rgb_path, ir_path, image_path_fus):
    rgb_image = Image.open(rgb_path).convert("RGB")
    ir_image = Image.open(ir_path).convert("L")

    ycbcr_image = rgb_image.convert("YCbCr")
    y_channel, _, _ = ycbcr_image.split()
    img_array = np.array(rgb_image)
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    rgb_tensor = transform(y_channel).unsqueeze(0).to(device)
    ir_tensor = transform(ir_image).unsqueeze(0).to(device)

    with torch.no_grad():
        test_vi_features = encoder(rgb_tensor)
        test_ir_features = encoder(ir_tensor)

        x_c = cfmd(test_vi_features, test_ir_features)
        x_fus = 0.1 * (x_c) + 0.9 * (test_vi_features + test_ir_features)
        generated_vi_images = decoder(x_fus)

        ones_1 = torch.ones_like(generated_vi_images)
        zeros_1 = torch.zeros_like(generated_vi_images)
        generated_vi_images = torch.where(generated_vi_images > ones_1, ones_1, generated_vi_images)
        generated_vi_images = torch.where(generated_vi_images < zeros_1, zeros_1, generated_vi_images)
        vi_img_pred = generated_vi_images.cpu().detach().numpy().squeeze()
        vi_img_pred = np.uint8(255.0 * vi_img_pred)
        img_hsv[:, :, 2] = vi_img_pred
        modifited_vi_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        modifited_vi_img = Image.fromarray(modifited_vi_img)
        modifited_vi_img.save(image_path_fus)

image_files = [f for f in os.listdir(dir_vi_test) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# Process each image
for filename in image_files:
    rgb_file_path = os.path.join(dir_vi_test, filename)
    ir_file_path = os.path.join(dir_ir_test, filename)
    fus_path = os.path.join(fus_directory, filename)
# Check if both files exist
    if os.path.exists(rgb_file_path) and os.path.exists(ir_file_path):
        try:
            fusion_images(rgb_file_path, ir_file_path, fus_path)
            print(f"Successfully processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    else:
        print(f"Skipping {filename} - one or both source files not found")
print("Image fusion process completed!")
