import torch
import numpy as np
import os
import sys
import json
import pathlib
from typing import List, Dict, Optional
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from PIL import Image

ROOT_DIR = os.getcwd()
sys.path.insert(0, ROOT_DIR)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import logging
logger = logging.getLogger(__name__)


DEVICE = os.getenv('DEVICE', 'cuda')
MODEL_CONFIG = os.getenv('MODEL_CONFIG')
MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT')

if DEVICE == 'cuda':
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# build path to the model checkpoint
sam2_checkpoint = str(os.path.join(ROOT_DIR, "", MODEL_CHECKPOINT))

sam2_model = build_sam2(MODEL_CONFIG, sam2_checkpoint, device=DEVICE)

print("SAM2 model loaded. CUDA available:", torch.cuda.is_available())
print("SAM2 model device:", next(sam2_model.parameters()).device)

predictor = SAM2ImagePredictor(sam2_model)

# -----------------------------------------------------------------------------
# DummyLabelInterface
# -----------------------------------------------------------------------------
class DummyLabelInterface:
    """
    DummyLabelInterface provides a minimal implementation of the label interface
    required by the base class (LabelStudioMLBase). Its primary role is to supply
    a default implementation for get_first_tag_occurrence, so that the model does not fail
    if a full label configuration is not provided during initialization.
    """
    def get_first_tag_occurrence(self, from_name="BrushLabels", to_name="Image", **kwargs):
        # Returns the provided names and a default key ("image") to access image URL from the task data.
        return from_name, to_name, "image"

# -----------------------------------------------------------------------------
# Sam2Nuclio Model
# -----------------------------------------------------------------------------
class Sam2Nuclio(LabelStudioMLBase):
    """
    Sam2Nuclio is a custom ML backend model for the SAM2 segmentation.
    It inherits from LabelStudioMLBase and is intended for deployment on Nuclio.
    
    This model:
      - Loads the SAM2 model and predictor.
      - Uses a dummy label interface if one is not provided.
      - Implements the predict method to generate segmentation masks based on input tasks and context.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, 'config') or self.config is None:
            self.config = {}
        if self.get("model_version") is None:
            self.set("model_version", "v1.0")
        if not hasattr(self, "label_interface"):
            self.label_interface = DummyLabelInterface()

    def get_results(self, masks, probs, width, height, from_name, to_name, label):
        """
        Process the raw masks and scores from the predictor into a list of annotations 
        that Label Studio understands (using RLE encoding).
        """
        results = []
        total_prob = 0
        for mask, prob in zip(masks, probs):
            # creates a random ID for your label everytime so no chance for errors
            label_id = str(uuid4())[:4]
            # converting the mask from the model to RLE format which is usable in Label Studio
            mask = mask * 255
            rle = brush.mask2rle(mask)
            total_prob += prob
            results.append({
                'id': label_id,
                'from_name': from_name,
                'to_name': to_name,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': [label],
                },
                'score': prob,
                'type': 'brushlabels',
                'readonly': False
            })

        return [{
            'result': results,
            'model_version': self.get('model_version'),
            'score': total_prob / max(len(results), 1)
        }]

    def set_image(self, image_url, task_id):
        """
        Loads the image from a URL (using get_local_path) and sets it on the predictor.
        """
        image_path = get_local_path(image_url, task_id=task_id)
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        predictor.set_image(image)

    def _sam_predict(self, img_url, point_coords=None, point_labels=None, input_box=None, task=None):
        """
        Calls the SAM2 predictor with the given prompt (keypoints, labels, etc.) and returns
        the segmentation mask and associated probability.
        """
        logger.info("CUDA available: " + str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            logger.info("Model device: " + str(next(sam2_model.parameters()).device))

        self.set_image(img_url, task.get('id'))
        point_coords = np.array(point_coords, dtype=np.float32) if point_coords else None
        point_labels = np.array(point_labels, dtype=np.float32) if point_labels else None
        input_box = np.array(input_box, dtype=np.float32) if input_box else None

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box,
            multimask_output=True
        )

        logger.info(f"Masks shape: {masks.shape if masks is not None else 'None'}, Scores: {scores}")

        if masks is None or masks.shape[0] == 0:
            logger.warning("No masks generated, returning empty response")
            return {'masks': [], 'probs': []}

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]

        mask = masks[0, :, :].astype(np.uint8)
        prob = float(scores[0])

        return {
            'masks': [mask],
            'probs': [prob]
        }

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """
        Returns the predicted segmentation mask for a given task and annotation context.
        If no context is provided, returns an empty prediction.
        """
        from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')

        if not context or not context.get('result'):
            # if there is no context, no interaction has happened yet
            return ModelResponse(predictions=[])

        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']

        # collect context information
        point_coords = []
        point_labels = []
        input_box = None
        selected_label = None
        for ctx in context['result']:
            x = ctx['value']['x'] * image_width / 100
            y = ctx['value']['y'] * image_height / 100
            ctx_type = ctx['type']
            selected_label = ctx['value'][ctx_type][0]
            if ctx_type == 'keypointlabels':
                point_labels.append(int(ctx.get('is_positive', 0)))
                point_coords.append([int(x), int(y)])
            elif ctx_type == 'rectanglelabels':
                box_width = ctx['value']['width'] * image_width / 100
                box_height = ctx['value']['height'] * image_height / 100
                input_box = [int(x), int(y), int(box_width + x), int(box_height + y)]

        print(f'Point coords are {point_coords}, point labels are {point_labels}, input box is {input_box}')

        img_url = tasks[0]['data'][value]
        predictor_results = self._sam_predict(
            img_url=img_url,
            point_coords=point_coords or None,
            point_labels=point_labels or None,
            input_box=input_box,
            task=tasks[0]
        )

        predictions = self.get_results(
            masks=predictor_results['masks'],
            probs=predictor_results['probs'],
            width=image_width,
            height=image_height,
            from_name=from_name,
            to_name=to_name,
            label=selected_label)
        
        return ModelResponse(predictions=predictions)

    def reload_model(self, checkpoint_path: str):
        """
        Reloads the SAM2 model from the given checkpoint.
        """
        global sam2_model, predictor
        sam2_model = build_sam2(MODEL_CONFIG, checkpoint_path, device=DEVICE)
        predictor = SAM2ImagePredictor(sam2_model)
        logger.info(f"Model reloaded from checkpoint: {checkpoint_path}")

    def find_latest_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """
        Finds the latest checkpoint in the given directory based on modification time.
        Returns the path to the latest checkpoint, or None if not found.
        """
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        if not checkpoint_files:
            return None
        latest = max(checkpoint_files, key=os.path.getmtime)
        return latest

    def process_training_data(self, training_data: List[Dict], output_dir: str) -> str:
        """
        Processes training data from Label Studio and writes it to a JSON file in the expected format.
        For a real SAM2 training pipeline, this function would convert the LS annotations into the format 
        required by the SAM2 training scripts.
        
        Returns the path to the created training data file.
        """
        training_file = os.path.join(output_dir, "training_data.json")
        with open(training_file, "w") as f:
            json.dump(training_data, f)
        logger.info(f"Training data written to {training_file}")
        return training_file

    def train(self, training_data: List[Dict], **kwargs) -> Dict:
        """
        Fine-tunes the SAM2 model using training data provided by Label Studio.
        This method:
          1. Processes the training data.
          2. Invokes the SAM2 training script (using subprocess) with the processed data.
          3. Finds the latest checkpoint from the training run.
          4. Reloads the model with the new checkpoint and updates the model version.
        
        Parameters:
          training_data (List[Dict]): A list of LS annotations or prediction results.
          kwargs: Additional parameters for training.
        
        Returns:
          dict: A dictionary with the new model version and training status.
        """
        import tempfile
        import subprocess

        logger.info("Starting SAM2 training with provided training data...")
        # Create a temporary directory to store training data and checkpoints.
        training_dir = tempfile.mkdtemp(prefix="sam2_training_")
        training_file = self.process_training_data(training_data, training_dir)
        checkpoint_dir = os.path.join(training_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Build the training command.
        # This command should be adjusted to match the SAM2 training pipeline.
        # For example, assume we have a training script "train.py" in the sam2 training folder.
        config_file = os.getenv("SAM2_TRAINING_CONFIG")  # e.g., path to a Hydra config file
        if config_file is None:
            raise Exception("SAM2_TRAINING_CONFIG environment variable is not set.")
        command = [
            "python", "-m", "train",  # assumes training module is invoked this way
            "--config-name", config_file,
            f"training_data_file={training_file}",
            f"checkpoint_dir={checkpoint_dir}"
        ]
        logger.info("Training command: " + " ".join(command))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("SAM2 training failed: " + result.stderr)
            raise Exception("SAM2 training failed")
        logger.info("SAM2 training completed successfully.")
        latest_checkpoint = self.find_latest_checkpoint(checkpoint_dir)
        if not latest_checkpoint:
            raise Exception("No checkpoint found after training.")
        self.reload_model(latest_checkpoint)
        new_version = "v1.1"  # Alternatively, parse the output or checkpoint filename to set a version.
        self.set("model_version", new_version)
        logger.info(f"Training complete. New model version: {new_version}")
        return {"model_version": new_version, "status": "training_complete", "checkpoint": latest_checkpoint}