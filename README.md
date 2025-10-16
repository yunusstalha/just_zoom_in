# Just Zoom In

## Goal
The project implements a sequential cross-view geo-localization framework that learns to iteratively zoom into large satellite maps to localize a ground-view image. A dual-stream ViT encoder pair extracts global and local representations from ground and satellite imagery, while an auto-regressive GPT-style decoder learns a zooming policy that selects the next map patch until the final location is identified.


## Repository Layout
```
configs/
  base.py             
data/
  dataset.py           # PyTorch Dataset for (ground image, satellite map, zoom sequence)
  transforms.py        # Augmentation and preprocessing pipelines
models/
  encoders.py          
  decoder.py           
  model.py             
utils/            
train.py               
eval.py               
README.md
```
