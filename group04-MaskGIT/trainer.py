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

random.seed(796)



def train(args):

	if args.device is None:
		args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print("No device is selected! Switching to {}".format(args.device))


	batch_size = args.batch_size
	epochs = args.epochs

	transformList = transforms.Compose([
		transforms.ToTensor(),
	 ])

	transform_to_image = transforms.ToPILImage()

	train_set = ImageDirectory(args.dataset_path, transform=transformList)
	train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
	

	model = MaskGIT(args).to(args.device)

	optimizer = torch.optim.Adam(model.transformer.parameters())

	loss_func = None
	losses = []

	for epoch in range(epochs):
		acc_loss = 0
		counter = 0
		for batch in tqdm(train_loader):
			optimizer.zero_grad()
			batch = batch.to(args.device)
			
			logits, labels = model(batch)

			loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

			loss.backward()
			
			acc_loss += loss.item() 
			#print(loss.item())
			optimizer.step()
			counter += 1

		acc_loss = acc_loss/(counter+1)
		print(acc_loss)
		losses.append(acc_loss)

		if (epoch+1) % args.ckpt_interval == 0:
			torch.save(model.state_dict(), "{}/epoch_{}_model.pt".format(MODELPATH, epoch))
			plt.plot(losses)
			plt.xlabel("Epoch")
			plt.ylabel("Loss")
			plt.title("Loss vs. Epoch graph")
			plt.savefig("figs/loss_{}.png".format(epoch))



def main(args):
	train(args)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("--epochs", type=int, default=300, help="Number of epochs for training")
	parser.add_argument("--batch_size", type=int, default=256, help="Size of batches for training")
	parser.add_argument("--ckpt_interval", type=int, default=1, help="Model save intervals")
	parser.add_argument("--dim", type=int, default=768)
	parser.add_argument("--hidden_dim", type=int, default=3072)
	parser.add_argument("--n_layers", type=int, default=24)
	parser.add_argument("--num_codebook_vectors", type=int, default=1024)
	parser.add_argument("--num_img_tok", type=int, default=256)
	parser.add_argument("--mask_token", type=int, default=1024)
	parser.add_argument("--model_path", help="Path of the pretrained model")
	parser.add_argument("--iters", type=int, default=8, help="Number of iterations for inference loop")
	parser.add_argument("--device", default=None)
	parser.add_argument("--dataset_path", default="Data/Imagenet64")


	args = parser.parse_args()


	main(args)