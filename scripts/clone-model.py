'''
clones models from Hugging Face to models/model-name.

Example:
python clone-model.py facebook/opt-1.3b

this seems nicer than the previous script https://github.com/oobabooga/text-generation-webui/blob/main/download-model.py

'''

from git import Repo
import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('MODEL', type=str, default=None, help="`tloen/alpaca-lora-7b`")
parser.add_argument('-b', '--branch', type=str, default='main', help='Name of the Git branch to download from.')
parser.add_argument('-t', '--text_only', action='store_true')
args = parser.parse_args()


if __name__ == '__main__':
    model = args.MODEL
    repo = 'https://huggingface.co/' + model
    name = model.replace('/', '_')
    output_folder = './data/loras' if 'lora' in name else './data/models'
    dest = f'{output_folder}/{name}_git'
    
    if args.text_only:
        os.environ['GIT_LFS_SKIP_SMUDGE']="1"
        

    result = subprocess.run(['git', 'lfs'], capture_output=True)
    assert result.returncode==0, 'git lfs should be installed'
        
    print(f'cloning "{repo}" to "{dest}"')
    Repo.clone_from(repo, dest, multi_options=['--depth=1', '--branch=main', '--filter=blob:none'])
    
    # FIXME it clones the git folder that has a huge size, it seems to have the old git lfs objects. so it's double the size. I can't seem to stop that
