import torch
import torchvision.transforms as T
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image
import pandas as pd
from glob import glob
import os
import time
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AGICConfig:
    """Configuration for AGIC implementation"""
    image_size: Tuple[int, int] = (224, 224)
    model_name: str = "Salesforce/blip2-opt-2.7b"
    max_new_tokens: int = 20
    num_beams: int = 5
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50

class AGICAttentionProcessor:
    """Processor for handling attention maps in AGIC"""
    def __init__(self, model: Blip2ForConditionalGeneration, device: torch.device):
        self.model = model
        self.device = device

    def extract_attention_maps(self, image_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention maps from vision model"""
        self.model.vision_model.config.output_attentions = True
        with torch.no_grad():
            outputs = self.model.vision_model(pixel_values=image_tensor)
        attentions = outputs.attentions
        maps = []
        for attn in attentions:
            # Calculate mean attention across heads, excluding CLS token
            attn_map = attn[0, :, 0, 1:].mean(0).reshape(16, 16)
            # Normalize attention map
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
            attn_map = T.functional.resize(attn_map.unsqueeze(0), [224, 224], antialias=True)[0]
            maps.append(attn_map)
        return maps

    def amplify_tensor_with_attention(self, image_tensor: torch.Tensor, attn_map: torch.Tensor) -> torch.Tensor:
        """Amplify image tensor using attention map"""
        attn_map = attn_map.to(image_tensor.device)
        amplified = image_tensor.clone()
        for c in range(3):
            amplified[0, c] = amplified[0, c] * (1 + attn_map)
        return torch.clamp(amplified, -1.0, 1.0)

class AGICCaptionGenerator:
    """Main AGIC caption generator class"""
    def __init__(self, config: Optional[AGICConfig] = None):
        self.config = config or AGICConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initializing model and processor
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.config.model_name, torch_dtype=torch.float16
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        
        # Setting up image transformation pipeline
        self.transform = T.Compose([
            T.Resize(self.config.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.attention_processor = AGICAttentionProcessor(self.model, self.device)
        
        logger.info(f"Initialized AGIC with model {self.config.model_name}")

    def load_image(self, image_path: str) -> Tuple[Image.Image, torch.Tensor]:
        """Load and preprocess image"""
        pil_img = Image.open(image_path).convert("RGB").resize(self.config.image_size)
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
        return pil_img, tensor_img

    def tensor_to_image(self, image_tensor: torch.Tensor) -> Image.Image:
        """Convert tensor back to PIL image"""
        unnorm = T.Compose([
            T.Normalize(mean=[0., 0., 0.], std=[1/0.5, 1/0.5, 1/0.5]),
            T.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.]),
            T.ToPILImage()
        ])
        return unnorm(image_tensor.squeeze(0).cpu())

    def generate_caption(self, image: Union[torch.Tensor, Image.Image]) -> str:
        """Generate caption for an image"""
        if isinstance(image, torch.Tensor):
            image = self.tensor_to_image(image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                num_beams=self.config.num_beams,
                max_new_tokens=self.config.max_new_tokens,
                early_stopping=True,
                num_return_sequences=1,
                do_sample=True,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                top_k=self.config.top_k
            )
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def process_image(self, image_path: str) -> Dict[str, str]:
        """Process single image and generate captions for different attention layers"""
        logger.info(f"Processing image: {image_path}")
        
        image_name = os.path.basename(image_path)
        pil_img, tensor_img = self.load_image(image_path)
        attn_maps = self.attention_processor.extract_attention_maps(tensor_img)

        # Generate captions for different attention layers
        first_layer_tensor = self.attention_processor.amplify_tensor_with_attention(tensor_img, attn_maps[0])
        middle_layer_tensor = self.attention_processor.amplify_tensor_with_attention(tensor_img, attn_maps[len(attn_maps)//2])
        last_layer_tensor = self.attention_processor.amplify_tensor_with_attention(tensor_img, attn_maps[-1])

        # Calculate max and mean attention maps
        attn_tensor = torch.stack(attn_maps)
        max_map = torch.max(attn_tensor, dim=0)[0]
        mean_map = torch.mean(attn_tensor, dim=0)

        max_tensor = self.attention_processor.amplify_tensor_with_attention(tensor_img, max_map)
        mean_tensor = self.attention_processor.amplify_tensor_with_attention(tensor_img, mean_map)

        # Generate captions
        result = {
            'image_name': image_name,
            'original_caption': self.generate_caption(pil_img),
            'first_layer': self.generate_caption(first_layer_tensor),
            'middle_layer': self.generate_caption(middle_layer_tensor),
            'last_layer': self.generate_caption(last_layer_tensor),
            'max_attention': self.generate_caption(max_tensor),
            'mean_attention': self.generate_caption(mean_tensor)
        }
        return result

class AGICEvaluator:
    """Evaluator for computing metrics across generated captions"""
    def __init__(self):
        self.scorers = {
            'BLEU': Bleu(n=4),
            'CIDEr': Cider(),
            'METEOR': Meteor(),
            'ROUGE': Rouge(),
            'SPICE': Spice()
        }

    def evaluate_captions(self, gts: Dict, res: Dict, layer_name: str) -> Dict[str, float]:
        """Evaluate captions for a specific layer"""
        logger.info(f"Evaluating captions for {layer_name}")
        results = {}
        for metric, scorer in self.scorers.items():
            score, _ = scorer.compute_score(gts, {k: v[layer_name] for k, v in res.items()})
            if isinstance(score, list):
                score = score[-1]  # Take BLEU-4 for BLEU
            results[metric] = float(score)
        return results

def run_agic_evaluation(image_dir: str, gts_path: str, output_path: str):
    """Main function to run AGIC evaluation"""
    config = AGICConfig()
    generator = AGICCaptionGenerator(config)
    evaluator = AGICEvaluator()

    # Load ground truth
    with open(gts_path, 'r') as f:
        gts = json.load(f)

    # Process images
    images = glob(os.path.join(image_dir, '*.jpg'))
    results = {}
    start_time = time.time()

    for image_path in images:
        result = generator.process_image(image_path)
        image_name = result['image_name']
        results[image_name] = {
            'original_caption': [result['original_caption']],
            'first_layer': [result['first_layer']],
            'middle_layer': [result['middle_layer']],
            'last_layer': [result['last_layer']],
            'max_attention': [result['max_attention']],
            'mean_attention': [result['mean_attention']]
        }

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Evaluate all layers
    layers = ['original_caption', 'first_layer', 'middle_layer', 'last_layer', 'max_attention', 'mean_attention']
    evaluation_results = {}
    
    for layer in layers:
        evaluation_results[layer] = evaluator.evaluate_captions(gts, results, layer)

    # Save evaluation results
    evaluation_output_path = output_path.replace('.json', '_metrics.json')
    with open(evaluation_output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    return evaluation_results

def main():
    """Entry point for AGIC evaluation"""
    image_dir = '/home/ubuntu/flickr8k/Images/'
    gts_path = '/content/gts.json'
    output_path = '/content/agic_8k.json'
    
    evaluation_results = run_agic_evaluation(image_dir, gts_path, output_path)
    
    # Print evaluation results
    print(json.dumps(evaluation_results, indent=2))

if __name__ == "__main__":
    main()