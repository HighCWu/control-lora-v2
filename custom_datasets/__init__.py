import random
import numpy as np

import torch
import torch.utils.data as data


class CustomDataset(data.Dataset):
    def __init__(self, dataset, args, tokenizer):
        self.dataset = dataset
        self.args = args
        self.image_column = args.image_column
        self.conditioning_image_column = args.conditioning_image_column
        self.caption_column = args.caption_column
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        pixel_values = data[self.image_column]
        conditioning_pixel_values = data[self.conditioning_image_column]
        input_ids = self.tokenize_captions(data[self.caption_column])
        if pixel_values.shape[-1] <= 4:
            pixel_values = torch.tensor(pixel_values).permute(2,0,1)
        if conditioning_pixel_values.shape[-1] <= 4:
            conditioning_pixel_values = torch.tensor(conditioning_pixel_values).permute(2,0,1)
        
        return dict(
            pixel_values=pixel_values,
            conditioning_pixel_values=conditioning_pixel_values,
            input_ids=input_ids,
        )
    
    def tokenize_captions(self, caption, is_train=True):
        captions = []
        if random.random() < self.args.proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{self.caption_column}` should contain either strings or lists of strings."
            )
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids[0]
