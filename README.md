Diffusion Model (DDPM) From Scratch – PyTorch

This project implements a Denoising Diffusion Probabilistic Model (DDPM) completely from scratch inside a single Jupyter Notebook. It includes the full forward noising process, reverse denoising sampling, a custom U-Net, and end-to-end training on the Stanford Cars dataset.

🚀 Overview

Full DDPM pipeline implemented step-by-step

Linear beta schedule + closed-form forward diffusion

Custom U-Net with sinusoidal timestep embeddings

Training on ~16k Stanford Cars images (HF dataset)

Sampling from pure noise → generating car images

Visualizations for both noising and denoising processes

📁 Contents

Everything is inside one notebook:

Data loading & preprocessing

Forward diffusion (noise scheduler)

U-Net model

Loss function & optimization

Reverse sampling + plotting

Training loop with sample outputs

🛠 Requirements
pip install torch torchvision datasets matplotlib numpy pillow

▶️ Running the Notebook

Simply open the notebook and run all cells:

jupyter notebook

📊 Results

The notebook displays:

Forward diffusion steps (image → full noise)

Reverse sampling steps (noise → generated cars)

Generated samples throughout training
