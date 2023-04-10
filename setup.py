from setuptools import find_packages, setup

setup(
    name='alpaca_convert',
    packages=find_packages(),
    version='0.1',
    description='Convert alpaca lora models to ggml, gptq, and non lora hf models',
)
