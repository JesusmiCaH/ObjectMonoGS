from romatch import roma_outdoor, roma_indoor
from PIL import Image
import numpy as np
import torch

roma_model = roma_indoor(device="cuda:0")
roma_model.upsample_preds = False
roma_model.symmetric = False

# Load two images
image1_path = "./datasets/bonn/rgbd_bonn_balloon_tracking/rgb/1548266633.53451.png"
image2_path = "./datasets/bonn/rgbd_bonn_balloon_tracking/rgb/1548266634.53747.png"

imA = Image.open(image1_path)
imB = Image.open(image2_path)

W,H = imA.size
print(H,W)

warp, certainty_warp = roma_model.match(imA, imB, device="cuda:0")
matches, certainty = roma_model.sample(warp, certainty_warp)


certainty = certainty_warp.reshape(-1).clone()
certainty[certainty > 0.8] = 1
good_samples = torch.multinomial(certainty, 10000, replacement=False)
matches_NN = warp.reshape(-1,4)[good_samples]

kptsA = matches_NN[:, :2]
kptsB = matches_NN[:, 2:]

kptsA_pix = torch.zeros_like(kptsA)
kptsB_pix = torch.zeros_like(kptsB)
kptsA_pix[:,0] = ((kptsA[:,0] + 1) * (W - 1) / 2)
kptsA_pix[:,1] = ((kptsA[:,1] + 1) * (H - 1) / 2)

kptsB_pix[:,0] = ((kptsB[:,0] + 1) * (W - 1) / 2)
kptsB_pix[:,1] = ((kptsB[:,1] + 1) * (H - 1) / 2)

kptsA_idx = np.round(kptsA_pix.detach().cpu().numpy()).astype(np.int32)
kptsB_idx = np.round(kptsB_pix.detach().cpu().numpy()).astype(np.int32)

import matplotlib.pyplot as plt

# Combine images A and B into a single plot and connect corresponding keypoints
combined_width = W * 2
combined_height = H

# Create a blank canvas for the combined image
combined_image = Image.new('RGB', (combined_width, combined_height))
combined_image.paste(imA, (0, 0))
combined_image.paste(imB, (W, 0))

plt.figure(figsize=(15, 7))
plt.imshow(combined_image)


print("somewhatsomehow")
print(kptsB_idx.shape)
# Adjust keypoints for the combined image
kptsB_idx_shifted = kptsB_idx.copy()
kptsB_idx_shifted[:, 0] += W  # Shift keypoints B horizontally

# Plot and connect keypoints
for i in range(0, 50):  # Example: plotting a subset of keypoints
    plt.plot(
        [kptsA_idx[i, 0], kptsB_idx_shifted[i, 0]],
        [kptsA_idx[i, 1], kptsB_idx_shifted[i, 1]],
        'g-', alpha=0.6
    )
    plt.scatter(kptsA_idx[i, 0], kptsA_idx[i, 1], c='r', s=10, alpha=0.8)
    plt.scatter(kptsB_idx_shifted[i, 0], kptsB_idx_shifted[i, 1], c='b', s=10, alpha=0.8)

plt.axis('off')
plt.title("Keypoints and Matches")
plt.savefig("keypoints_combined.png", dpi=300)