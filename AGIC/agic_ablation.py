import torch
import torchvision.transforms as T
from PIL import Image
import os
from glob import glob
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import time

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AGICConfig:
    """Configuration for AGIC implementation with ablation studies"""
    image_size: Tuple[int, int] = (224, 224)
    model_name: str = "Salesforce/blip2-opt-2.7b"
    max_new_tokens: int = 20
    num_beams: int = 5
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    amplification_factors: List[int] = (1, 3, 5, 10)
    image_dir: str = "/home/ubuntu/flickr8k/Images/"
    gts_path: str = "/content/gts.json"
    output_path: str = "/content/agic_ablation.json"

class AGICAttentionProcessor:
    """Processor for handling attention maps in AGIC"""
    def __init__(self, model: Blip2ForConditionalGeneration, device: torch.device):
        self.model = model
        self.device = device
        logger.info("Initialized AGICAttentionProcessor for BLIP-2")

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

    def amplify_tensor_with_attention(self, image_tensor: torch.Tensor, attn_map: torch.Tensor, k: int) -> torch.Tensor:
        """Amplify image tensor using attention map with amplification factor k"""
        attn_map = attn_map.to(image_tensor.device)
        amplified = image_tensor.clone()
        for c in range(3):
            amplified[0, c] = amplified[0, c] * (1 + k * attn_map)
        return torch.clamp(amplified, -1.0, 1.0)

class AGICCaptionGenerator:
    """Main AGIC caption generator class with ablation support"""
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

    def generate_caption(self, image: Union[torch.Tensor, Image.Image], decoding_strategy: str) -> str:
        """Generate caption for an image using specified decoding strategy"""
        if isinstance(image, torch.Tensor):
            image = self.tensor_to_image(image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        
        generate_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "num_return_sequences": 1
        }
        
        if decoding_strategy == "topk":
            generate_kwargs.update({
                "do_sample": True,
                "top_k": self.config.top_k,
                "temperature": self.config.temperature
            })
        elif decoding_strategy == "topp":
            generate_kwargs.update({
                "do_sample": True,
                "top_p": self.config.top_p,
                "temperature": self.config.temperature
            })
        elif decoding_strategy == "beamsearch":
            generate_kwargs.update({
                "num_beams": self.config.num_beams,
                "early_stopping": True
            })
        else:  # all
            generate_kwargs.update({
                "num_beams": self.config.num_beams,
                "do_sample": True,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "temperature": self.config.temperature,
                "early_stopping": True
            })

        with torch.no_grad():
            out = self.model.generate(**inputs, **generate_kwargs)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def process_image(self, image_path: str) -> Dict[str, Dict[str, str]]:
        """Process single image and generate captions for different attention layers and decoding strategies"""
        logger.info(f"Processing image: {image_path}")
        
        image_name = os.path.basename(image_path)
        pil_img, tensor_img = self.load_image(image_path)
        attn_maps = self.attention_processor.extract_attention_maps(tensor_img)

        result = {}
        decoding_strategies = ["topk", "topp", "beamsearch", "all"]
        
        for decoding_strategy in decoding_strategies:
            strategy_result = {
                'image_name': image_name,
                'original_caption': self.generate_caption(pil_img, decoding_strategy)
            }
            
            # Generate captions for different attention layers and amplification factors
            for k in self.config.amplification_factors:
                # First layer
                first_layer_tensor = self.attention_processor.amplify_tensor_with_attention(tensor_img, attn_maps[0], k)
                strategy_result[f'first_layer_k{k}'] = self.generate_caption(first_layer_tensor, decoding_strategy)
                
                # Middle layer
                middle_layer_tensor = self.attention_processor.amplify_tensor_with_attention(tensor_img, attn_maps[len(attn_maps)//2], k)
                strategy_result[f'middle_layer_k{k}'] = self.generate_caption(middle_layer_tensor, decoding_strategy)
                
                # Last layer
                last_layer_tensor = self.attention_processor.amplify_tensor_with_attention(tensor_img, attn_maps[-1], k)
                strategy_result[f'last_layer_k{k}'] = self.generate_caption(last_layer_tensor, decoding_strategy)
                
                # Max attention
                attn_tensor = torch.stack(attn_maps)
                max_map = torch.max(attn_tensor, dim=0)[0]
                max_tensor = self.attention_processor.amplify_tensor_with_attention(tensor_img, max_map, k)
                strategy_result[f'max_attention_k{k}'] = self.generate_caption(max_tensor, decoding_strategy)
                
                # Mean attention
                mean_map = torch.mean(attn_tensor, dim=0)
                mean_tensor = self.attention_processor.amplify_tensor_with_attention(tensor_img, mean_map, k)
                strategy_result[f'mean_attention_k{k}'] = self.generate_caption(mean_tensor, decoding_strategy)
            
            result[decoding_strategy] = strategy_result
        
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

    def evaluate_captions(self, gts: Dict, res: Dict, caption_key: str) -> Dict[str, float]:
        """Evaluate captions for a specific configuration"""
        logger.info(f"Evaluating captions for {caption_key}")
        results = {}
        for metric, scorer in self.scorers.items():
            score, _ = scorer.compute_score(gts, {k: [v[caption_key]] for k, v in res.items()})
            if isinstance(score, list):
                score = score[-1]  # Take BLEU-4 for BLEU
            results[metric] = float(score)
        return results

def run_agic_evaluation(image_dir: str, gts_path: str, output_path: str):
    """Main function to run AGIC evaluation with ablations"""
    config = AGICConfig(image_dir=image_dir, gts_path=gts_path, output_path=output_path)
    generator = AGICCaptionGenerator(config)
    evaluator = AGICEvaluator()

    # Load ground truth
    with open(gts_path, 'r') as f:
        gts = json.load(f)

    # Process images
    images = glob(os.path.join(image_dir, '*.jpg'))
    results = {}
    start_time = time.time()

    for image_path in tqdm(images, desc="Processing Images"):
        result = generator.process_image(image_path)
        image_name = result['topk']['image_name']
        results[image_name] = {}
        
        for decoding_strategy in result:
            results[image_name][decoding_strategy] = {
                'original_caption': [result[decoding_strategy]['original_caption']],
                **{f'first_layer_k{k}': [result[decoding_strategy][f'first_layer_k{k}']] for k in config.amplification_factors},
                **{f'middle_layer_k{k}': [result[decoding_strategy][f'middle_layer_k{k}']] for k in config.amplification_factors},
                **{f'last_layer_k{k}': [result[decoding_strategy][f'last_layer_k{k}']] for k in config.amplification_factors},
                **{f'max_attention_k{k}': [result[decoding_strategy][f'max_attention_k{k}']] for k in config.amplification_factors},
                **{f'mean_attention_k{k}': [result[decoding_strategy][f'mean_attention_k{k}']] for k in config.amplification_factors}
            }

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Evaluate all configurations
    evaluation_results = {}
    decoding_strategies = ["topk", "topp", "beamsearch", "all"]
    
    for decoding_strategy in decoding_strategies:
        evaluation_results[decoding_strategy] = {}
        caption_keys = ['original_caption'] + \
                       [f'{layer}_k{k}' for k in config.amplification_factors 
                        for layer in ['first_layer', 'middle_layer', 'last_layer', 'max_attention', 'mean_attention']]
        
        for caption_key in caption_keys:
            evaluation_results[decoding_strategy][caption_key] = evaluator.evaluate_captions(gts, results, f"{decoding_strategy}/{caption_key}")

    # Save evaluation results
    evaluation_output_path = output_path.replace('.json', '_metrics.json')
    with open(evaluation_output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    return evaluation_results

def main():
    """Entry point for AGIC evaluation with ablations"""
    image_dir = '/home/ubuntu/flickr8k/Images/'
    gts_path = '/content/gts.json'
    output_path = '/content/agic_ablation.json'
    
    evaluation_results = run_agic_evaluation(image_dir, gts_path, output_path)
    
    # Print evaluation results
    print(json.dumps(evaluation_results, indent=2))

if __name__ == "__main__":
    main()