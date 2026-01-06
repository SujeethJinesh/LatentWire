#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test that the training module can be imported successfully."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test core imports
        from latentwire import train
        print("✓ latentwire.train imported successfully")
        
        from latentwire import models
        print("✓ latentwire.models imported successfully")
        
        from latentwire import losses
        print("✓ latentwire.losses imported successfully")
        
        from latentwire import core_utils
        print("✓ latentwire.core_utils imported successfully")
        
        from latentwire import data
        print("✓ latentwire.data imported successfully")
        
        from latentwire import checkpointing
        print("✓ latentwire.checkpointing imported successfully")
        
        # Test that key classes are available
        from latentwire.models import InterlinguaInterlinguaEncoder, Adapter, LMWrapper
        print("✓ Key model classes available")
        
        from latentwire.losses import k_token_ce_from_prefix, kd_first_k_prefix_vs_text
        print("✓ Key loss functions available")
        
        print("\nAll imports successful! The training module is ready to use.")
        return True
        
    except ImportError as e:
        print("✗ Import failed: {}".format(e))
        print("\nPlease install required dependencies:")
        print("  pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
