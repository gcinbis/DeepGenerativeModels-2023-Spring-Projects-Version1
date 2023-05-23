import math
import torch
import torch.nn as nn
from VQGAN.taming_transformers.vqImporter import vqImporter
from models.models import BidirectionalTransformer
import numpy as np
from maskgit_utils import utils


class MaskGIT(nn.Module):
	def __init__(self, args):
		super().__init__()

		self.args = args
		self.tokenizer = vqImporter().eval()
		self.transformer = BidirectionalTransformer(args)

	def forward(self, x):

		with torch.no_grad():
			quants, _, (_, _, min_encoding_indices) = self.tokenizer.encode(x)

		min_encoding_indices = min_encoding_indices.view(quants.shape[0], -1)

		orgs = min_encoding_indices.clone()

		# get the mask
		mask = self.create_mask(min_encoding_indices.shape[1])

		# Use broadcasting to apply the mask to all instances in the batch
		mask = mask.unsqueeze(0) # add a singleton dimension to align the dimensions for broadcasting
		mask = mask.expand(min_encoding_indices.shape[0], -1) # expand the mask to the size of the batch

		# Apply mask, if m = 0, do not change, if m = 1, mask it!
		min_encoding_indices[~mask] = self.args.mask_token


		sos_token = torch.full(size=(quants.shape[0], 1), fill_value=self.args.mask_token+1, dtype=torch.long, device=self.args.device)


		min_encoding_indices = torch.cat((sos_token, min_encoding_indices), dim=1)
		orgs = torch.cat((sos_token, orgs), dim=1)

		logits = self.transformer(min_encoding_indices)

		return logits, orgs


	def create_mask(self, sequence_length):
		r = utils.cosine_scheduler(np.random.uniform())		
		num_tokens_to_mask = math.ceil(r * sequence_length) # get the # of tokens to mask

		mask = torch.zeros(sequence_length, dtype=torch.bool) # Initialize a mask with all False

		# get the indices to be masked
		mask_indices = torch.randperm(sequence_length)[:num_tokens_to_mask]
		
		# set these indices to True in mask tensor
		mask[mask_indices] = True

		return mask


	def get_mask_inference(self, confidence_scores, t, T):
		ratio = (t+1) / T
		num_tokens_to_mask = math.ceil( utils.cosine_scheduler(ratio) * confidence_scores.shape[1]) 

		# Create the initial mask
		mask = torch.zeros_like(confidence_scores, dtype=torch.bool)

		# Sort the tokens by their confidence scores
		sorted_indices = torch.argsort(confidence_scores, dim=-1)

		# Get the indices of the tokens with the lowest confidence scores
		mask_indices = sorted_indices[:, :num_tokens_to_mask]

		# Set these indices to True in the mask tensor

		mask[:, mask_indices] = True

		return mask




	def inpaint_image(self, input_images, iterations=8):
		# Get empty canvas for input images
		with torch.no_grad():
			quants, _, (_, _, input_tokens) = self.tokenizer.encode(input_images)
		input_tokens = input_tokens.view(quants.shape[0], -1)

		# Generate SOS tokens and concatenate with the encoded images
		sos_tokens = torch.full(size=(quants.shape[0], 1), fill_value=self.args.mask_token+1, dtype=torch.long).to(input_tokens.device)
		input_tokens = torch.cat([sos_tokens, input_tokens], dim=1)

		# Get the number of unknown pixels
		unknown_pixels = torch.sum(input_tokens == self.args.mask_token, dim=-1)

		for t in range(iterations):
			break
			# Get logits from the transformer
			logits = self.transformer(input_tokens)

			# Calculate probabilities
			probs = nn.Softmax(dim=-1)(logits)

			# Sample ids with their probability score
			sampled_ids = torch.distributions.categorical.Categorical(probs).sample()

			# Calculate confidence scores
			confidence_scores = torch.gather(probs, -1, sampled_ids.unsqueeze(-1)).squeeze(-1)

			# Get the mask using dynamic masking strategy
			mask = self.get_mask_inference(confidence_scores, t, iterations)

			# Update current ids, mask tokens with lower confidence
			input_tokens = torch.where(mask, self.args.mask_token, sampled_ids)

		# Return the filled canvas


		vectors = self.tokenizer.quantize.embedding(input_tokens[:, 1:]).reshape(input_tokens.shape[0], 4, 4, 256)

		vectors = vectors.permute(0, 3, 1, 2)

		out_images = self.tokenizer.decode(vectors)

		print(out_images.shape)

		return out_images