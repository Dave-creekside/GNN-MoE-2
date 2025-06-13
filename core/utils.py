#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py

Core utility functions for the MoE framework.
"""

import os
import warnings

def suppress_warnings():
    """Suppresses common, non-critical warnings to clean up output."""
    
    # Suppress NotOpenSSLWarning from urllib3
    try:
        from urllib3.exceptions import NotOpenSSLWarning
        warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
    except ImportError:
        pass

    # Suppress tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Suppress other common warnings if needed
    warnings.filterwarnings("ignore", "is_categorical_dtype")
    warnings.filterwarnings("ignore", "is_numeric_dtype")
    warnings.filterwarnings("ignore", "use_inf_as_na")
