'''
clones models from Hugging Face to models/model-name.

Example:
python clone-model.py facebook/opt-1.3b

'''

from git import Repo
import argparse
from tqdm.auto import tqdm
from git import RemoteProgress


parser = argparse.ArgumentParser()
parser.add_argument('MODEL', type=str, default=None, help="`tloen/alpaca-lora-7b`")
parser.add_argument('--branch', type=str, default='main', help='Name of the Git branch to download from.')
args = parser.parse_args()

class CloneProgress(RemoteProgress):
    """tqdm progress bar for GitPython"""
    def __init__(self):
        super().__init__()
        self.pbar = tqdm()

    def update(self, op_code, cur_count, max_count=None, message=''):
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.refresh()

if __name__ == '__main__':
    model = args.MODEL
    repo = 'https://huggingface.co/' + model
    name = model.replace('/', '_')
    dest = f'./models/{name}'
    print(f'cloning "{repo}" to "{dest}"')
    Repo.clone_from(repo, dest, progress=CloneProgress())
