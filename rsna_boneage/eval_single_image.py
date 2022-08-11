from boneage.rsna_bone_litmodel import RSNABoneageLitModel
from boneage.rsna_bone_net import resnet18 as resnet18_boneage
from boneage.rsna_bone_dataloading import RSNABoneageDataset, get_image_transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import numpy as np
from PIL import Image


use_path = True
image_path = './images/train_val/1377.png'
checkpoint_file = './lightning_logs/9/checkpoints/epoch=9-val_loss=241.238510.ckpt'

resnet = resnet18_boneage(pretrained=False, progress=True, num_classes=1)
model = RSNABoneageLitModel.load_from_checkpoint(checkpoint_file, net=resnet, mc_iterations=50)

# Load given image or random image from test set
if use_path:
    print(f'Loading given image "{image_path}"')
    # Load image from disk
    image = Image.open(image_path).convert('RGB')
    # Resize and convert to Tensor
    transform = get_image_transforms()
    image: torch.Tensor = transform(image)
    # Add "dummy" batch dimension
    image = image.unsqueeze(dim=0)
# else:
#     print('Loading random image from test set...')
#     data = DogVsCatDataset('./test_annotations.csv')
#     test_dataloader = DataLoader(data, batch_size=1, shuffle=True)
#     batch = next(iter(test_dataloader))
#     image: torch.Tensor = batch[0]

model.cuda()
image = image.cuda()
pred, metrics = model.forward_with_mc(image)
predictions = metrics['predictions'].cpu().detach().numpy()

print('PREDICTED BONE AGE:', pred.detach().numpy())

plt.figure()
plt.imshow(image.cpu()[0].squeeze().permute(1, 2, 0))
plt.show(block=False)

plt.figure()
plt.hist(predictions, bins=50, range=[0, 250])
plt.show(block=False)

plt.figure()
plt.boxplot(predictions)
plt.ylim(0, 250)
plt.show(block=False)

del metrics['predictions']
for key, val in metrics.items():
    print(f'{key}: {val}')

plt.show()
