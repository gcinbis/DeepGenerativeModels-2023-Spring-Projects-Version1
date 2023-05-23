import sys
sys.path.append(".")

from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# also disable grad to save memory
import torch
#torch.set_grad_enabled(False) # BAD BAD THING TO DO

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import yaml
import torch
from omegaconf import OmegaConf
from VQGAN.taming_transformers.taming.models.vqgan import VQModel, GumbelVQ




def load_config(config_path, display=False):
	config = OmegaConf.load(config_path)
	if display:
		print(yaml.dump(OmegaConf.to_container(config)))
	return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
	if is_gumbel:
		model = GumbelVQ(**config.model.params)
	else:
		model = VQModel(**config.model.params)
	if ckpt_path is not None:
		sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
		missing, unexpected = model.load_state_dict(sd, strict=False)
	return model.eval()

def preprocess_vqgan(x):
	x = 2.*x - 1.
	return x

def custom_to_pil(x):
	x = x.detach().cpu()
	x = torch.clamp(x, -1., 1.)
	x = (x + 1.)/2.
	x = x.permute(1,2,0).numpy()
	x = (255*x).astype(np.uint8)
	x = Image.fromarray(x)
	if not x.mode == "RGB":
		x = x.convert("RGB")
	return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
	z, _, [_, _, indices] = model.encode(x)
	print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
	xrec = model.decode(z)
	return xrec

def preprocess(img, target_image_size=256, map_dalle=True):
	s = min(img.size)
	
	if s < target_image_size:
		raise ValueError(f'min dim for image {s} < {target_image_size}')
		
	r = target_image_size / s
	s = (round(r * img.size[1]), round(r * img.size[0]))
	img = TF.resize(img, s, interpolation=Image.LANCZOS)
	img = TF.center_crop(img, output_size=2 * [target_image_size])
	img = torch.unsqueeze(T.ToTensor()(img), 0)
	if map_dalle: 
	  img = map_pixels(img)
	return img



def vqImporter():
	config1024 = load_config("VQGAN/pretrain/vqgan_imagenet_f16_1024/configs/model.yaml", display=False)
	model1024 = load_vqgan(config1024, ckpt_path="VQGAN/pretrain/vqgan_imagenet_f16_1024/ckpts/last.ckpt").to(DEVICE)

	return model1024