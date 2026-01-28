#!/usr/bin/env python3
"""
Unified segmentation pipeline CLI.
Single entry point for all segmentation models.
"""
import argparse
import sys
import os

# CRITICAL: Ensure source directory is in python path BEFORE any local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import shutil
import numpy as np

from models.adapters.base_adapter import SegmentationResult
from models.adapters.openyolo3d_adapter import OpenYolo3DAdapter
from models.adapters.mask3d_adapter import Mask3DAdapter
from utils_source.vocabulary import load_vocabulary
from utils_source.export_utils.output_manager import OutputManager

# Output functions moved to OutputManager class

import datetime
import logging

def setup_logger(output_dir):
    """Setup logging to file and stdout."""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"segmentation_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    return logging.getLogger(__name__), log_file

def main():
    parser = argparse.ArgumentParser(
        description='Run 3D segmentation with OpenYOLO3D or Mask3D',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_segmentation.py --data data/pipeline_output --model openyolo3d --vocab furniture
        """
    )
    
    parser.add_argument('--data', required=True, help='Path to data directory containing "export" folder')
    parser.add_argument('--model', choices=['openyolo3d', 'mask3d'], help='Model to use (overrides config)')
    parser.add_argument('--vocab', help='Vocabulary mode: lvis, coco, furniture (overrides config)')
    parser.add_argument('--config', help='Path to custom config file')
    parser.add_argument('--frame-step', type=int, help='Frame step for OpenYOLO3D (overrides config)')
    parser.add_argument('--conf-threshold', type=float, help='Confidence threshold (overrides config)')
    
    args = parser.parse_args()
    
    # Path Derivation
    data_dir = os.path.abspath(args.data)
    scene_dir = os.path.join(data_dir, 'export')
    
    # Validation
    if not os.path.isdir(scene_dir):
        print(f"[ERROR] Export directory not found: {scene_dir}")
        print(f"       Expected structure: {args.data}/export/")
        sys.exit(1)
        
    # Load config (early load for model name defaults)
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../configs/segmentation.yaml')
        
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Apply CLI overrides to dict before validation
    if args.model:
        config_dict['model']['name'] = args.model
    if args.vocab:
        config_dict['vocabulary']['mode'] = args.vocab
    if args.frame_step:
        config_dict.setdefault('inference', {})['frame_step'] = args.frame_step
    if args.conf_threshold:
        config_dict.setdefault('inference', {})['conf_threshold'] = args.conf_threshold
    
    # Validate configuration with Pydantic
    from models.config_models import load_and_validate_config
    try:
        validated_config = load_and_validate_config(config_dict)
        # Convert back to dict for compatibility with existing code
        config = validated_config.model_dump()
    except Exception as e:
        print(f"[ERROR] Configuration validation failed: {e}")
        return 1
        
    # Determine model name for output folder
    model_name = config['model']['name']
    output_dir = os.path.join(data_dir, f"{model_name}_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Logger
    logger, log_file = setup_logger(data_dir)
    logger.info(f"Starting Segmentation Pipeline")
    logger.info(f"Data Directory: {data_dir}")
    logger.info(f"Input Scene:    {scene_dir}")
    logger.info(f"Output Dir:     {output_dir}")
    logger.info(f"Log File:       {log_file}")
    logger.info(f"Configuration validated successfully")
    
    # Log applied overrides
    if args.model:
        logger.info(f"Override model: {args.model}")
    if args.vocab:
        logger.info(f"Override vocabulary: {args.vocab}")
    if args.frame_step:
        logger.info(f"Override frame_step: {args.frame_step}")
    if args.conf_threshold:
        logger.info(f"Override conf_threshold: {args.conf_threshold}")
        
    # Initialize adapter
    if model_name == 'openyolo3d':
        adapter = OpenYolo3DAdapter(config)
    elif model_name == 'mask3d':
        adapter = Mask3DAdapter(config)
    else:
        logger.error(f"Unknown model: {model_name}")
        return 1
    
    logger.info(f"Initialized {model_name} adapter")
    
    # Vocabulary
    vocab_mode = config['vocabulary']['mode']
    custom_classes = config['vocabulary'].get('custom_classes', None)
    vocabulary = load_vocabulary(vocab_mode, custom_classes)
    
    logger.info(f"Using vocabulary: {vocab_mode} ({len(vocabulary)} classes)")
    
    # Run Inference
    try:
        # Capture stdout/stderr to redirect library output to logger
        import io
        import contextlib
        
        # Create string buffers  
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        logger.info("="*60)
        logger.info("Starting model inference (capturing all output)...")
        logger.info("="*60)
        
        # Capture all print statements from the library
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            result = adapter.predict(scene_dir, vocabulary, output_dir)
        
        # Log captured output
        stdout_content = stdout_buffer.getvalue()
        stderr_content = stderr_buffer.getvalue()
        
        if stdout_content:
            logger.info("="*60)
            logger.info("Captured stdout from model:")
            logger.info("="*60)
            for line in stdout_content.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        
        if stderr_content:
            logger.warning("="*60)
            logger.warning("Captured stderr from model:")
            logger.warning("="*60)
            for line in stderr_content.split('\n'):
                if line.strip():
                    logger.warning(f"  {line}")
        
        logger.info("="*60)
        logger.info("Segmentation complete!")
        logger.info(f"Instances detected: {len(result.scores)}")
        logger.info("="*60)
        
        # Save all outputs using OutputManager
        output_mgr = OutputManager(output_dir, config)
        output_mgr.save_all(result, vocabulary)
            
        # Note: Scene Graph generation removed per user request.
        # It is now a separate step to be run on the data folder.
            
        return 0
    except Exception as e:
        logger.error(f"Segmentation failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
