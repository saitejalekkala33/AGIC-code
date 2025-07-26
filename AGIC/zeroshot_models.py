import time
import torch
from PIL import Image
import os
from glob import glob
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration, FuyuProcessor, FuyuForCausalLM, LlavaForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ZeroShotConfig:
    """Configuration for zero-shot caption generation"""
    image_dir: str = "/home/ubuntu/flickr8k/Images/"
    gts_path: str = "/content/gts.json"
    output_path: str = "/content/zeroshot_captions.json"
    max_new_tokens: int = 30
    blip2_model: str = "Salesforce/blip2-opt-2.7b"
    llava_model: str = "llava-hf/llava-1.5b-7b" 
    qwen_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    fuyu_model: str = "adept/fuyu-8b"
    batch_size: int = 1

class Blip2Captioner:
    """BLIP-2 zero-shot caption generator"""
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        logger.info(f"Initialized BLIP-2 model: {model_name}")

    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption using BLIP-2"""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=30)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

class LlavaCaptioner:
    """LLaVA zero-shot caption generator"""
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        logger.info(f"Initialized LLaVA model: {model_name}")

    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption using LLaVA"""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Generate a COCO-style caption"
                    "The caption should be a single, grammatically correct English sentence, "
                    "focused on the main objects and their actions or interactions. "
                    "Avoid using proper names, and make the caption informative but concise. "
                    "Use the style typically seen in the MS COCO dataset."}
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device, dtype=torch.float16 if v.dtype == torch.float32 else v.dtype)
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_new_tokens=30)
        return self.processor.batch_decode(generate_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0].strip()


class QwenCaptioner:
    """Qwen zero-shot caption generator"""
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        logger.info(f"Initialized Qwen model: {model_name}")

    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption using Qwen"""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Generate a COCO-style caption"
                    "The caption should be a single, grammatically correct English sentence, "
                    "focused on the main objects and their actions or interactions. "
                    "Avoid using proper names, and make the caption informative but concise. "
                    "Use the style typically seen in the MS COCO dataset."}
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device, dtype=torch.float16 if v.dtype == torch.float32 else v.dtype)
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_new_tokens=30)
        return self.processor.batch_decode(generate_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0].strip()

class FuyuCaptioner:
    """Fuyu zero-shot caption generator"""
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model = FuyuForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        self.processor = FuyuProcessor.from_pretrained(model_name)
        logger.info(f"Initialized Fuyu model: {model_name}")

    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption using Fuyu"""
        text_prompt = "Generate a descriptive, COCO-style caption for this image. The caption should be a single, grammatically correct English sentence, focused on the main objects and their actions or interactions. Avoid using proper names, and make the caption informative but concise. Use the style typically seen in the MS COCO dataset.\n"
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=30)
        return self.processor.batch_decode(outputs[:, -30:], skip_special_tokens=True)[0].strip()

class ZeroShotEvaluator:
    """Evaluator for computing metrics for zero-shot captions"""
    def __init__(self):
        self.scorers = {
            'BLEU': Bleu(n=4),
            'CIDEr': Cider(),
            'METEOR': Meteor(),
            'ROUGE': Rouge(),
            'SPICE': Spice()
        }

    def evaluate_captions(self, gts: Dict, res: Dict, model_name: str) -> Dict[str, float]:
        """Evaluate captions for a specific model"""
        logger.info(f"Evaluating captions for {model_name}")
        results = {}
        for metric, scorer in self.scorers.items():
            score, _ = scorer.compute_score(gts, {k: v[model_name] for k, v in res.items()})
            if isinstance(score, list):
                score = score[-1]  # Take BLEU-4 for BLEU
            results[metric] = float(score)
        return results

class ZeroShotCaptionGenerator:
    """Main class for zero-shot caption generation and evaluation"""
    def __init__(self, config: Optional[ZeroShotConfig] = None):
        self.config = config or ZeroShotConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize captioners
        self.captioners = {
            'blip2': Blip2Captioner(self.config.blip2_model, self.device),
            'llava': LlavaCaptioner(self.config.llava_model, self.device),
            'qwen': QwenCaptioner(self.config.qwen_model, self.device),
            'fuyu': FuyuCaptioner(self.config.fuyu_model, self.device)
        }
        self.evaluator = ZeroShotEvaluator()
        logger.info("Initialized ZeroShotCaptionGenerator")

    def process_images(self) -> Dict:
        """Process images and generate captions for all models"""
        images = glob(os.path.join(self.config.image_dir, '*.jpg'))
        results = {}
        
        for image_path in tqdm(images, desc="Processing Images"):
            image_name = os.path.basename(image_path)
            image = Image.open(image_path).convert("RGB")
            results[image_name] = {}
            
            for model_name, captioner in self.captioners.items():
                caption = captioner.generate_caption(image)
                results[image_name][model_name] = [caption]
        
        return results

    def run_evaluation(self) -> Dict:
        """Run caption generation and evaluation for all models"""
        logger.info("Starting zero-shot caption generation and evaluation")
        
        # Load ground truth
        with open(self.config.gts_path, 'r') as f:
            gts = json.load(f)
        
        # Generate captions
        start_time = time.time()
        results = self.process_images()
        
        # Save captions
        with open(self.config.output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Evaluate all models
        evaluation_results = {}
        for model_name in self.captioners.keys():
            evaluation_results[model_name] = self.evaluator.evaluate_captions(gts, results, model_name)
        
        # Save evaluation results
        evaluation_output_path = self.config.output_path.replace('.json', '_metrics.json')
        with open(evaluation_output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        end_time = time.time()
        logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
        
        return evaluation_results

def main():
    """Entry point for zero-shot caption evaluation"""
    config = ZeroShotConfig()
    generator = ZeroShotCaptionGenerator(config)
    evaluation_results = generator.run_evaluation()
    
    # Print evaluation results
    print(json.dumps(evaluation_results, indent=2))

if __name__ == "__main__":
    main()