import argparse
from os.path import join

import torch
from torchvision import transforms
from PIL import Image

def preprocess(data_dir, split):
  assert split in ["train", "validate", "test"]

  print("Process {} dataset...".format(split))
  image_dir = join(data_dir, "formula_images_processed")

  formulas_file = join(data_dir, "im2latex_formulas.norm.lst")
  with open(formulas_file, 'r') as f:
    formulas = [x.strip('\n') for x in f.readlines()]

  split_file = join(data_dir, "im2latex_{}_filter.lst".format(split))
  pairs = []
  transform = transforms.ToTensor()
  with open(split_file, 'r') as f:
    for line in f:
      img_name, formula_id = line.strip('\n').split()
      # load img and its corresponding formula
      img_tensor = transform(Image.open(join(image_dir, img_name)))
      formula = formulas[int(formula_id)]
      pairs.append((img_tensor, formula))
    pairs.sort(key=img_size)

  out_file = join(data_dir, "{}.pkl".format(split))
  torch.save(pairs, out_file)
  print("Save {} dataset to {}".format(split, out_file))

def img_size(pair):
  img, formula = pair
  return tuple(img.size())

if __name__ == "__main__":
  parser = argparse.ArgumentParser( description="Im2Latex Data Preprocess Program")
  parser.add_argument("--data_path", type=str, default="./data/", help="The dataset's dir")
  args = parser.parse_args()
  splits = ["validate", "test", "train"]
  for s in splits:
    preprocess(args.data_path, s)
