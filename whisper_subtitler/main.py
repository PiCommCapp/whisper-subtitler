"""
Main entry point for the whisper-subtitler application.

This module provides the entry point for the CLI.
"""

import sys

from whisper_subtitler.modules.cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
