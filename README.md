# AGIC: Attention-Guided Image Captioning

## What is AGIC?
AGIC (Attention-Guided Image Captioning) is a framework designed to improve the relevance of image captions by leveraging a contextual relevance amplification mechanism, implemented through an attention-guided process. Inspired by recent research ([Liu et al., 2025](https://arxiv.org/abs/2501.09997)), AGIC uses attention patterns from vision transformers to amplify relevant image regions and generate more accurate and meaningful captions.

### Model Description
AGIC works in three main stages:
1. **Attention Weights Extraction:**
   - Extracts attention maps from a pre-trained vision transformer to identify the most relevant image regions.
2. **Image Amplification:**
   - Amplifies the original image features using the extracted attention weights, making relevant regions more prominent.
3. **Caption Generation:**
   - Generates captions for the amplified image using a hybrid decoding strategy (beam search, Top-k, Top-p sampling, temperature scaling) to enhance diversity and fluency.

## Datasets Used
- **Flickr8k** ([HuggingFace link](https://huggingface.co/datasets/jxie/flickr8k))
- **Flickr30k** ([HuggingFace link](https://huggingface.co/datasets/nlphuji/flickr30k))

These datasets provide images and multiple human-annotated captions for benchmarking image captioning models.

## How to Run
All scripts are located in the `AGIC/` directory. Each script can be run from the command line. You may need to adjust the default paths for your dataset and ground truth files.

### 1. AGIC with BLIP-2
Runs AGIC using the BLIP-2 model.
```bash
python AGIC/agic_blip2.py
```
- **Default paths:**
  - Images: `/home/ubuntu/flickr8k/Images/`
  - Ground truth: `/content/gts.json`
  - Output: `/content/agic_8k.json`
- To change paths, edit the variables in the `main()` function or pass them as arguments if you add argparse support.

### 2. AGIC with LLaVA
Runs AGIC using the LLaVA model.
```bash
python AGIC/agic_llava.py
```
- **Default paths:**
  - Images: `/home/ubuntu/flickr8k/Images/`
  - Ground truth: `/content/gts.json`
  - Output: `/content/agic_llava.json`

### 3. AGIC Ablation Study
Runs AGIC with ablation studies (different amplification factors and decoding strategies).
```bash
python AGIC/agic_ablation.py
```
- **Default paths:**
  - Images: `/home/ubuntu/flickr8k/Images/`
  - Ground truth: `/content/gts.json`
  - Output: `/content/agic_ablation.json`

### 4. Zero-Shot Captioning Baselines
Runs zero-shot captioning with BLIP-2, LLaVA, Qwen, and Fuyu models for comparison.
```bash
python AGIC/zeroshot_models.py
```
- **Default paths:**
  - Images: `/home/ubuntu/flickr8k/Images/`
  - Ground truth: `/content/gts.json`
  - Output: `/content/zeroshot_captions.json`

> **Note:** All scripts can be modified to accept command-line arguments for paths and parameters by adding `argparse` if needed.

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Example requirements.txt
```
torch
transformers
torchvision
pillow
pandas
tqdm
pycocoevalcap
```
- You may also need: `json`, `os`, `glob`, `logging`, `dataclasses` (Python 3.7+), and `typing` (standard library).
- For some models (e.g., BLIP-2, LLaVA, Qwen, Fuyu), you may need to install additional model-specific dependencies from HuggingFace.

## Model Description (Technical)
AGIC extracts attention weights from a vision transformer, amplifies the image features using these weights, and generates captions using a hybrid decoding strategy. See the code and comments for mathematical details and implementation.

---

## References
- [Flickr8k Dataset](https://huggingface.co/datasets/jxie/flickr8k)
- [Flickr30k Dataset](https://huggingface.co/datasets/nlphuji/flickr30k)
- Liu et al., 2025. Attention-guided Self-reflection for Zero-shot Hallucination Detection in Large Language Models
