import argparse
import pickle as pkl
from os.path import join
from collections import Counter

class Vocab(object):
  def __init__(self):
    self.sign_to_id = {"<s>": START_TOKEN, "</s>": END_TOKEN, "<pad>": PAD_TOKEN, "<unk>": UNK_TOKEN}
    self.id_to_sign = dict((index, token) for token, index in self.sign_to_id.items())
    self.length = 4

  def add_sign(self, sign):
    if sign not in self.sign_to_id:
      self.sign_to_id[sign] = self.length
      self.id_to_sign[self.length] = sign
      self.length += 1

  def __len__(self):
    return self.length

def build_vocab(data_dir, min_count=10):
  """
  traverse training formulas to make vocab
  and store the vocab in the file
  """
  vocab = Vocab()
  counter = Counter()

  formulas_file = join(data_dir, 'im2latex_formulas.norm.lst')
  with open(formulas_file, 'r') as f:
    formulas = [x.strip('\n') for x in f.readlines()]

  with open(join(data_dir, 'im2latex_train_filter.lst'), 'r') as f:
    for line in f:
      _, index = line.strip('\n').split()
      index = int(index)
      formula = formulas[index].split()
      counter.update(formula)

  for word, count in counter.most_common():
    if count >= min_count:
      vocab.add_sign(word)
  vocab_file = join(data_dir, 'vocab.pkl')
  print("Writing Vocab File in ", vocab_file)
  with open(vocab_file, 'wb') as w:
    pkl.dump(vocab, w)

def load_vocab(data_dir):
  with open(join(data_dir, 'vocab.pkl'), 'rb') as f:
    vocab = pkl.load(f)
  print("Load vocab including {} words!".format(len(vocab)))
  return vocab

if __name__ == "__main__":
  START_TOKEN, PAD_TOKEN, END_TOKEN, UNK_TOKEN = 0, 1, 2, 3
  parser = argparse.ArgumentParser(description="Building vocab for Im2Latex")
  parser.add_argument("--data_path", type=str, default="./data/", help="The dataset's dir")
  args = parser.parse_args()
  vocab = build_vocab(args.data_path)
