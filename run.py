#!/usr/bin/env python
"""
Simple runner script for whisper-subtitler.
"""

import os
import sys

# Add the whisper-subtitler directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "whisper-subtitler"))

from modules.cli import main

if __name__ == "__main__":
    sys.exit(main())
