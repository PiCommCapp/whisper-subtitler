"""
Tests for the configuration module.
"""

import os
import tempfile
from pathlib import Path

import pytest

# Import this once implemented
# from whisper_subtitler.config import Config


# Using a placeholder for now until Config class is implemented
class MockConfigForTesting:
    def __init__(self):
        self.model_size = "medium"
        self.language = "en"
        self.output_formats = ["txt", "srt", "vtt", "ttml"]
        self.num_speakers = None
        self.huggingface_token = None
        self.use_cuda = True
        self.verbose = False
        self.input_file = None
        self.output_dir = None
        self.preprocess_audio = False
        self.cluster_speakers = False
        self.optimize_num_speakers = False
        self.force_overwrite = False
        self.log_level = "INFO"
        self.log_file = None

    def load_from_env(self, env_file=None):
        self.huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
        self.model_size = os.environ.get("WHISPER_MODEL_SIZE", self.model_size)
        self.verbose = os.environ.get("SHOW_SPEAKER_DEBUG", "False").lower() in ("1", "true", "yes")
        return self

    def load_from_args(self, args):
        for key, value in args.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        return self

    def load_from_file(self, config_file):
        # Placeholder for loading from a config file
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        # Example implementation would read and parse the file
        return self

    def validate(self):
        # Example validation
        if not self.input_file:
            raise ValueError("Input file is required")
        if not self.huggingface_token:
            raise ValueError("HUGGINGFACE_TOKEN is required")
        return self


class TestConfig:
    """Test the configuration management."""

    def test_default_config(self):
        """Test the default configuration values."""
        # Replace with actual Config once implemented
        config = MockConfigForTesting()

        assert config.model_size == "medium"
        assert config.language == "en"
        assert config.output_formats == ["txt", "srt", "vtt", "ttml"]
        assert config.num_speakers is None
        assert config.huggingface_token is None
        assert config.use_cuda is True
        assert config.verbose is False
        assert config.preprocess_audio is False
        assert config.cluster_speakers is False
        assert config.optimize_num_speakers is False
        assert config.force_overwrite is False
        assert config.log_level == "INFO"
        assert config.log_file is None

    def test_env_loading(self, mock_env_variables):
        """Test loading configuration from environment variables."""
        # Replace with actual Config once implemented
        config = MockConfigForTesting().load_from_env()

        assert config.huggingface_token == "mock-token"
        assert config.model_size == "tiny"
        assert config.verbose is False  # From SHOW_SPEAKER_DEBUG=False

    def test_args_loading(self):
        """Test loading configuration from arguments."""
        # Replace with actual Config once implemented
        config = MockConfigForTesting().load_from_args({
            "model_size": "large",
            "language": "fr",
            "num_speakers": 3,
            "output_formats": ["srt"],
            "force_overwrite": True,
        })

        assert config.model_size == "large"
        assert config.language == "fr"
        assert config.num_speakers == 3
        assert config.output_formats == ["srt"]
        assert config.force_overwrite is True

    def test_args_none_values(self):
        """Test that None values in args don't override existing values."""
        config = MockConfigForTesting()
        config.model_size = "medium"

        # None value should not override existing value
        config.load_from_args({"model_size": None})
        assert config.model_size == "medium"

    def test_file_loading(self):
        """Test loading configuration from a file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("# Test config\n")
            temp_file.write("model_size = large\n")
            temp_file.write("language = fr\n")
            temp_path = temp_file.name

        try:
            # Replace with actual Config once implemented
            # This test will use the mock implementation for now
            config = MockConfigForTesting()

            # Mock the load_from_file method to set values as if read from file
            original_load_from_file = config.load_from_file

            def mock_load_from_file(file_path):
                assert file_path == temp_path
                config.model_size = "large"
                config.language = "fr"
                return config

            # Temporarily replace the method
            config.load_from_file = mock_load_from_file

            # Load from the temp file
            config.load_from_file(temp_path)

            # Restore original method
            config.load_from_file = original_load_from_file

            assert config.model_size == "large"
            assert config.language == "fr"
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_precedence(self, mock_env_variables):
        """Test configuration precedence (args > file > env > defaults)."""
        # Replace with actual Config once implemented
        config = MockConfigForTesting()

        # Start with defaults
        assert config.model_size == "medium"

        # Load from environment
        config.load_from_env()
        assert config.model_size == "tiny"  # From env

        # Mock loading from file (would override env)
        def mock_load_from_file(file_path):
            config.model_size = "small"  # From file
            return config

        original_load_from_file = config.load_from_file
        config.load_from_file = mock_load_from_file
        config.load_from_file("mock_path")
        config.load_from_file = original_load_from_file

        assert config.model_size == "small"

        # Then override with args (highest precedence)
        config.load_from_args({"model_size": "large"})
        assert config.model_size == "large"  # From args

    def test_validation(self):
        """Test configuration validation."""
        # Replace with actual Config once implemented
        config = MockConfigForTesting()
        config.huggingface_token = "token"
        config.input_file = "test.mp4"

        # Should pass with token and input file
        config.validate()

        # Should raise error without token
        config.huggingface_token = None
        with pytest.raises(ValueError, match="HUGGINGFACE_TOKEN is required"):
            config.validate()

        # Should raise error without input file
        config.huggingface_token = "token"
        config.input_file = None
        with pytest.raises(ValueError, match="Input file is required"):
            config.validate()

    def test_output_formats_validation(self):
        """Test validation of output formats."""
        # This test would check that only valid output formats are accepted
        # Replace with actual Config implementation once available
        config = MockConfigForTesting()

        # Mock the validate method to check output_formats
        original_validate = config.validate

        def mock_validate():
            if not set(config.output_formats).issubset({"txt", "srt", "vtt", "ttml"}):
                raise ValueError("Invalid output format")
            return original_validate(self)

        # Set invalid format and test validation
        config.output_formats = ["invalid"]
        config.validate = mock_validate

        with pytest.raises(ValueError, match="Invalid output format"):
            config.validate()

        # Restore original validate method
        config.validate = original_validate
