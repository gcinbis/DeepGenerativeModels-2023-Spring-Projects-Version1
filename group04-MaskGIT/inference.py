import torch

from models.MaskGIT import MaskGIT
#from maskgit_utils.ImageDirectory import ImageDirectory
from maskgit_utils.TinyImageNetDirectory import ImageDirectory
import numpy as np
from torchvision import transforms
import argparse
from tqdm import tqdm
import random
import matplotlib.pyplot as plt




MODELPATH = "SavedModels"


#np.random.seed(796)



def sample_from_dataset(dataset, n=1, mask_x=16, mask_y=16, mask_size=32):
	
	org_temp = []
	mask_temp = []

	mask = torch.zeros(size=(64, 64), dtype=bool)

	mask[mask_x:mask_x+mask_size, mask_y:mask_y+mask_size] = True

	mask = torch.stack((mask, mask, mask))
	print(mask.shape)


	for i in range(n):
		idx = np.random.randint(low=0, high=len(dataset))
		img = dataset[idx]

		masked = img.clone()
		masked[mask] = 0

		org_temp.append(img)
		mask_temp.append(masked)

	originals = torch.stack(org_temp)
	masked = torch.stack(mask_temp)

	return originals, masked

def infer_image(args):

	if args.device is None:
		args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print("No device is selected! Switching to {}".format(args.device))

	print(args.device)
	model = MaskGIT(args).to(args.device)

	model.load_state_dict(torch.load(args.model_path, map_location=args.device))

	model = model

	model.eval()


	transformList = transforms.Compose([
		transforms.ToTensor(),
	 ])
	tensortoImage = transforms.ToPILImage()
	dataset = ImageDirectory(args.dataset_path, transform=transformList)

	originals, masked = sample_from_dataset(dataset, args.samples)

	originals = originals.to(args.device)
	masked = masked.to(args.device)





	out_images = model.inpaint_image(masked, iterations=args.iters)


	for i in range(originals.shape[0]):
		org_img = tensortoImage(originals[i])
		masked_img = tensortoImage(masked[i])
		out_img = tensortoImage(out_images[i])

		fig, ax = plt.subplots(1, 3)


		ax[0].imshow(org_img)
		ax[0].set_title("Original Image")
		ax[1].imshow(masked_img)
		ax[1].set_title("Masked Image")
		ax[2].imshow(out_img)
		ax[2].set_title("Output of the Masked Image")

		plt.show()



def main(args):
	infer_image(args)


if __name__ == '__main__':


	parser = argparse.ArgumentParser()
	
	parser.add_argument("--model_path", help="Path of the pretrained model")
	parser.add_argument("--iters", type=int, default=8, help="Number of iterations for inference loop")
	parser.add_argument("--device", default=None)
	parser.add_argument("--samples", type=int, default=10, help="Number of samples to reconstruct")
	parser.add_argument("--epochs", type=int, default=300, help="Number of epochs for training")
	parser.add_argument("--batch_size", type=int, default=256, help="Size of batches for training")
	parser.add_argument("--ckpt_interval", type=int, default=1, help="Model save intervals")
	parser.add_argument("--dim", type=int, default=768)
	parser.add_argument("--hidden_dim", type=int, default=3072)
	parser.add_argument("--n_layers", type=int, default=24)
	parser.add_argument("--num_codebook_vectors", type=int, default=1024)
	parser.add_argument("--num_img_tok", type=int, default=256)
	parser.add_argument("--mask_token", type=int, default=1024)
	parser.add_argument("--dataset_path", default="Data/Imagenet64")


	args = parser.parse_args()

	main(args)