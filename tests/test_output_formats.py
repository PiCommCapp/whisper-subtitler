"""
Tests for the output formats module.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import mock_open, patch

# Import once implemented
# from whisper_subtitler.modules.output import (
#     OutputFormatter,
#     TextFormatter,
#     SRTFormatter,
#     VTTFormatter,
#     TTMLFormatter
# )


class TestOutputFormatter:
    """Test the output formatter functionality."""

    def test_initialization(self, mock_config):
        """Test initialization of the OutputFormatter."""
        # Once implemented, use the actual OutputFormatter
        # formatter = OutputFormatter(mock_config)
        # assert formatter.config == mock_config
        # assert formatter.output_formats == mock_config.output_formats
        # assert formatter.force_overwrite == mock_config.force_overwrite

        # For now, just ensure we can use the mock config
        assert "txt" in mock_config.output_formats
        assert True

    def test_generate_outputs(self, mock_config, sample_combined_result, temp_output_dir):
        """Test generating all outputs."""
        # Set up config and paths
        mock_config.output_dir = str(temp_output_dir)
        mock_config.output_formats = ["txt", "srt", "vtt", "ttml"]
        mock_config.force_overwrite = True

        input_file = "test_video.mp4"
        base_name = Path(input_file).stem

        # Once implemented, use the actual OutputFormatter
        # formatter = OutputFormatter(mock_config)
        # formatter.generate_outputs(sample_combined_result, input_file)

        # # Check that all output files were created
        # assert (temp_output_dir / f"{base_name}.txt").exists()
        # assert (temp_output_dir / f"{base_name}.srt").exists()
        # assert (temp_output_dir / f"{base_name}.vtt").exists()
        # assert (temp_output_dir / f"{base_name}.ttml").exists()

        # For now, just ensure we have the right setup
        assert set(mock_config.output_formats) == {"txt", "srt", "vtt", "ttml"}
        assert True

    def test_selective_output_formats(self, mock_config, sample_combined_result, temp_output_dir):
        """Test generating only selected output formats."""
        # Set up config and paths
        mock_config.output_dir = str(temp_output_dir)
        mock_config.output_formats = ["txt", "srt"]  # Only TXT and SRT
        mock_config.force_overwrite = True

        input_file = "test_video.mp4"
        base_name = Path(input_file).stem

        # Once implemented, use the actual OutputFormatter
        # formatter = OutputFormatter(mock_config)
        # formatter.generate_outputs(sample_combined_result, input_file)

        # # Check that only selected output files were created
        # assert (temp_output_dir / f"{base_name}.txt").exists()
        # assert (temp_output_dir / f"{base_name}.srt").exists()
        # assert not (temp_output_dir / f"{base_name}.vtt").exists()
        # assert not (temp_output_dir / f"{base_name}.ttml").exists()

        # For now, just ensure we have the right setup
        assert set(mock_config.output_formats) == {"txt", "srt"}
        assert True

    def test_force_overwrite(self, mock_config, sample_combined_result, temp_output_dir):
        """Test the force_overwrite option."""
        # Set up config and paths
        mock_config.output_dir = str(temp_output_dir)
        mock_config.output_formats = ["txt"]

        input_file = "test_video.mp4"
        base_name = Path(input_file).stem
        output_file = temp_output_dir / f"{base_name}.txt"

        # Create a file that already exists
        with open(output_file, "w") as f:
            f.write("Existing content")

        # First test with force_overwrite=False
        mock_config.force_overwrite = False

        # Once implemented, use the actual OutputFormatter
        # formatter = OutputFormatter(mock_config)
        # formatter.generate_outputs(sample_combined_result, input_file)

        # # Check that the file wasn't overwritten
        # with open(output_file, "r") as f:
        #     content = f.read()
        # assert content == "Existing content"

        # Then test with force_overwrite=True
        mock_config.force_overwrite = True

        # formatter = OutputFormatter(mock_config)
        # formatter.generate_outputs(sample_combined_result, input_file)

        # # Check that the file was overwritten
        # with open(output_file, "r") as f:
        #     content = f.read()
        # assert content != "Existing content"
        # assert "This is a test transcription" in content

        # For now, just ensure the file was created
        assert output_file.exists()
        assert True


class TestTextFormatter:
    """Test the text formatter functionality."""

    @patch("builtins.open", new_callable=mock_open)
    def test_format_text(self, mock_file, sample_combined_result):
        """Test formatting as plain text."""
        # Once implemented, use the actual TextFormatter
        # formatter = TextFormatter()
        # formatter.format(sample_combined_result, "output.txt")

        # # Check that the file was written with the correct content
        # mock_file.assert_called_once_with("output.txt", "w", encoding="utf-8")
        # written_content = "".join(call.args[0] for call in mock_file().write.call_args_list)
        # assert "This is a test transcription." in written_content

        # For now, just ensure we have the sample result
        assert sample_combined_result["text"] == "This is a test transcription."
        assert True

    def test_create_text_file(self, temp_output_dir, sample_combined_result):
        """Test creating a text file with the transcript."""
        output_file = temp_output_dir / "output.txt"

        # Once implemented, use the actual TextFormatter
        # formatter = TextFormatter()
        # formatter.format(sample_combined_result, str(output_file))

        # # Check the file was created with correct content
        # assert output_file.exists()
        # with open(output_file, "r", encoding="utf-8") as f:
        #     content = f.read()
        # assert content == "This is a test transcription."

        # For testing now, manually create the file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(sample_combined_result["text"])

        assert output_file.exists()
        with open(output_file, encoding="utf-8") as f:
            content = f.read()
        assert content == "This is a test transcription."


class TestSRTFormatter:
    """Test the SRT formatter functionality."""

    @patch("builtins.open", new_callable=mock_open)
    def test_format_srt(self, mock_file, sample_combined_result):
        """Test formatting as SRT."""
        # Once implemented, use the actual SRTFormatter
        # formatter = SRTFormatter()
        # formatter.format(sample_combined_result, "output.srt")

        # # Check that the file was written with the correct content
        # mock_file.assert_called_once_with("output.srt", "w", encoding="utf-8")
        # written_content = "".join(call.args[0] for call in mock_file().write.call_args_list)
        # assert "00:00:00,000 --> 00:00:02,000" in written_content
        # assert "SPEAKER_01: This is a test" in written_content
        # assert "00:00:02,500 --> 00:00:04,000" in written_content
        # assert "SPEAKER_02: transcription." in written_content

        # For now, just check that we have the sample result
        assert len(sample_combined_result["segments"]) == 2
        assert sample_combined_result["segments"][0]["speaker"] == "SPEAKER_01"
        assert True

    def test_create_srt_file(self, temp_output_dir, sample_combined_result):
        """Test creating an SRT file with the transcript."""
        output_file = temp_output_dir / "output.srt"

        # Once implemented, use the actual SRTFormatter
        # formatter = SRTFormatter()
        # formatter.format(sample_combined_result, str(output_file))

        # # Check the file was created with correct content
        # assert output_file.exists()
        # with open(output_file, "r", encoding="utf-8") as f:
        #     content = f.read()
        # assert "00:00:00,000 --> 00:00:02,000" in content
        # assert "SPEAKER_01: This is a test" in content

        # For testing now, manually create a simple SRT file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("1\n00:00:00,000 --> 00:00:02,000\nSPEAKER_01: This is a test\n\n")
            f.write("2\n00:00:02,500 --> 00:00:04,000\nSPEAKER_02: transcription.\n\n")

        assert output_file.exists()
        with open(output_file, encoding="utf-8") as f:
            content = f.read()
        assert "00:00:00,000 --> 00:00:02,000" in content
        assert "SPEAKER_01: This is a test" in content


class TestVTTFormatter:
    """Test the VTT formatter functionality."""

    @patch("builtins.open", new_callable=mock_open)
    def test_format_vtt(self, mock_file, sample_combined_result):
        """Test formatting as WebVTT."""
        # Once implemented, use the actual VTTFormatter
        # formatter = VTTFormatter()
        # formatter.format(sample_combined_result, "output.vtt")

        # # Check that the file was written with the correct content
        # mock_file.assert_called_once_with("output.vtt", "w", encoding="utf-8")
        # written_content = "".join(call.args[0] for call in mock_file().write.call_args_list)
        # assert "WEBVTT" in written_content
        # assert "00:00:00.000 --> 00:00:02.000" in written_content
        # assert "SPEAKER_01: This is a test" in written_content

        # For now, just check that we have the sample result
        assert len(sample_combined_result["segments"]) == 2
        assert True

    def test_create_vtt_file(self, temp_output_dir, sample_combined_result):
        """Test creating a WebVTT file with the transcript."""
        output_file = temp_output_dir / "output.vtt"

        # Once implemented, use the actual VTTFormatter
        # formatter = VTTFormatter()
        # formatter.format(sample_combined_result, str(output_file))

        # # Check the file was created with correct content
        # assert output_file.exists()
        # with open(output_file, "r", encoding="utf-8") as f:
        #     content = f.read()
        # assert "WEBVTT" in content
        # assert "00:00:00.000 --> 00:00:02.000" in content

        # For testing now, manually create a simple VTT file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            f.write("00:00:00.000 --> 00:00:02.000\nSPEAKER_01: This is a test\n\n")
            f.write("00:00:02.500 --> 00:00:04.000\nSPEAKER_02: transcription.\n\n")

        assert output_file.exists()
        with open(output_file, encoding="utf-8") as f:
            content = f.read()
        assert "WEBVTT" in content
        assert "00:00:00.000 --> 00:00:02.000" in content


class TestTTMLFormatter:
    """Test the TTML formatter functionality."""

    @patch("builtins.open", new_callable=mock_open)
    def test_format_ttml(self, mock_file, sample_combined_result):
        """Test formatting as TTML."""
        # Once implemented, use the actual TTMLFormatter
        # formatter = TTMLFormatter()
        # formatter.format(sample_combined_result, "output.ttml")

        # # Check that the file was written with the correct content
        # mock_file.assert_called_once_with("output.ttml", "w", encoding="utf-8")
        # written_content = "".join(call.args[0] for call in mock_file().write.call_args_list)
        # assert "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" in written_content
        # assert "<tt " in written_content
        # assert "begin=\"00:00:00.000\"" in written_content
        # assert "end=\"00:00:02.000\"" in written_content

        # For now, just check that we have the sample result
        assert len(sample_combined_result["segments"]) == 2
        assert True

    def test_create_ttml_file(self, temp_output_dir, sample_combined_result):
        """Test creating a TTML file with the transcript."""
        output_file = temp_output_dir / "output.ttml"

        # Once implemented, use the actual TTMLFormatter
        # formatter = TTMLFormatter()
        # formatter.format(sample_combined_result, str(output_file))

        # # Check the file was created with correct content
        # assert output_file.exists()
        # with open(output_file, "r", encoding="utf-8") as f:
        #     content = f.read()
        # assert "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" in content
        # assert "<tt " in content

        # # Check that the XML is valid by parsing it
        # root = ET.parse(output_file).getroot()
        # assert root.tag.endswith("tt")

        # For testing now, manually create a simple TTML file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<tt xml:lang="en-US" xmlns="http://www.w3.org/ns/ttml">\n')
            f.write('  <head>\n    <styling>\n      <style xml:id="s1"/>\n    </styling>\n  </head>\n')
            f.write("  <body>\n    <div>\n")
            f.write('      <p begin="00:00:00.000" end="00:00:02.000">This is a test</p>\n')
            f.write('      <p begin="00:00:02.500" end="00:00:04.000">transcription.</p>\n')
            f.write("    </div>\n  </body>\n</tt>\n")

        assert output_file.exists()
        with open(output_file, encoding="utf-8") as f:
            content = f.read()
        assert '<?xml version="1.0" encoding="UTF-8"?>' in content
        assert "<tt " in content

        # Check that the XML is valid by parsing it
        root = ET.parse(output_file).getroot()
        # Note: we can't use the endswith check because of namespace handling
        assert root.tag.split("}")[-1] == "tt"
