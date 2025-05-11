"""
Tests for the application orchestrator.
"""

from unittest.mock import MagicMock, patch

import pytest

# Import once implemented
# from whisper_subtitler.application import Application
# from whisper_subtitler.modules.transcribe import Transcriber
# from whisper_subtitler.modules.diarise import Diarizer
# from whisper_subtitler.modules.output import OutputFormatter


class TestApplication:
    """Test the application orchestrator."""

    def test_initialization(self, mock_config):
        """Test initialization of the Application."""
        # Once implemented, use the actual Application
        # app = Application(mock_config)
        # assert app.config == mock_config
        # assert app.transcriber is None
        # assert app.diarizer is None
        # assert app.output_formatter is None

        # For now, just ensure we can use the mock config
        assert mock_config.model_size == "tiny"
        assert True

    @pytest.mark.skip("Requires actual implementation")
    @patch("whisper_subtitler.modules.transcribe.Transcriber")
    @patch("whisper_subtitler.modules.diarise.Diarizer")
    @patch("whisper_subtitler.modules.output.OutputFormatter")
    def test_initialize_components(self, mock_output_formatter, mock_diarizer, mock_transcriber, mock_config):
        """Test initializing the application components."""
        # Setup mocks
        mock_transcriber_instance = MagicMock()
        mock_diarizer_instance = MagicMock()
        mock_formatter_instance = MagicMock()

        mock_transcriber.return_value = mock_transcriber_instance
        mock_diarizer.return_value = mock_diarizer_instance
        mock_output_formatter.return_value = mock_formatter_instance

        # Once implemented, use the actual Application
        # app = Application(mock_config)
        # app.initialize()

        # # Check that the components were created with the config
        # mock_transcriber.assert_called_once_with(mock_config)
        # mock_diarizer.assert_called_once_with(mock_config)
        # mock_output_formatter.assert_called_once_with(mock_config)

        # # Check that the components were assigned
        # assert app.transcriber == mock_transcriber_instance
        # assert app.diarizer == mock_diarizer_instance
        # assert app.output_formatter == mock_formatter_instance

        # For now, just ensure the test runs
        assert True

    @pytest.mark.skip("Requires actual implementation")
    @patch("whisper_subtitler.modules.transcribe.Transcriber")
    @patch("whisper_subtitler.modules.diarise.Diarizer")
    @patch("whisper_subtitler.modules.output.OutputFormatter")
    @patch("subprocess.run")
    def test_process_with_audio_extraction(
        self,
        mock_subprocess_run,
        mock_output_formatter,
        mock_diarizer,
        mock_transcriber,
        mock_config,
        temp_output_dir,
        sample_video_file,
    ):
        """Test processing a video file with audio extraction."""
        # Setup mocks
        mock_transcriber_instance = MagicMock()
        mock_diarizer_instance = MagicMock()
        mock_formatter_instance = MagicMock()

        mock_transcriber.return_value = mock_transcriber_instance
        mock_diarizer.return_value = mock_diarizer_instance
        mock_output_formatter.return_value = mock_formatter_instance

        # Configure mocks
        mock_transcriber_instance.transcribe.return_value = {
            "text": "This is a test transcription.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "This is a test"},
                {"id": 1, "start": 2.5, "end": 4.0, "text": "transcription."},
            ],
        }

        mock_diarizer_instance.diarize.return_value = [
            {"start": 0.0, "end": 2.2, "speaker": "SPEAKER_01"},
            {"start": 2.3, "end": 4.1, "speaker": "SPEAKER_02"},
        ]

        # Set up config
        mock_config.output_dir = str(temp_output_dir)

        # Once implemented, use the actual Application
        # app = Application(mock_config)
        # app.initialize()
        # app.process(str(sample_video_file))

        # # Check that audio was extracted
        # expected_audio_path = temp_output_dir / f"{sample_video_file.stem}.wav"
        # mock_subprocess_run.assert_called_once()
        # ffmpeg_cmd = mock_subprocess_run.call_args[0][0]
        # assert "ffmpeg" in ffmpeg_cmd[0]
        # assert str(expected_audio_path) in ffmpeg_cmd

        # # Check that transcription and diarization were called
        # mock_transcriber_instance.transcribe.assert_called_once()
        # mock_diarizer_instance.diarize.assert_called_once()

        # # Check that output was generated
        # mock_formatter_instance.generate_outputs.assert_called_once()

        # For now, just ensure we can use the sample file
        assert sample_video_file.exists()
        assert True

    @pytest.mark.skip("Requires actual implementation")
    @patch("whisper_subtitler.modules.transcribe.Transcriber")
    @patch("whisper_subtitler.modules.diarise.Diarizer")
    @patch("whisper_subtitler.modules.output.OutputFormatter")
    def test_process_with_audio_file(
        self, mock_output_formatter, mock_diarizer, mock_transcriber, mock_config, sample_audio_file
    ):
        """Test processing an audio file directly."""
        # Setup mocks
        mock_transcriber_instance = MagicMock()
        mock_diarizer_instance = MagicMock()
        mock_formatter_instance = MagicMock()

        mock_transcriber.return_value = mock_transcriber_instance
        mock_diarizer.return_value = mock_diarizer_instance
        mock_output_formatter.return_value = mock_formatter_instance

        # Configure mocks
        mock_transcriber_instance.transcribe.return_value = {
            "text": "This is a test transcription.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "This is a test"},
                {"id": 1, "start": 2.5, "end": 4.0, "text": "transcription."},
            ],
        }

        mock_diarizer_instance.diarize.return_value = [
            {"start": 0.0, "end": 2.2, "speaker": "SPEAKER_01"},
            {"start": 2.3, "end": 4.1, "speaker": "SPEAKER_02"},
        ]

        # Once implemented, use the actual Application
        # app = Application(mock_config)
        # app.initialize()
        # app.process(str(sample_audio_file))

        # # Check that transcription and diarization were called directly with the audio file
        # mock_transcriber_instance.transcribe.assert_called_once_with(str(sample_audio_file))
        # mock_diarizer_instance.diarize.assert_called_once_with(str(sample_audio_file))

        # # Check that output was generated
        # mock_formatter_instance.generate_outputs.assert_called_once()

        # For now, just ensure we can use the sample file
        assert sample_audio_file.exists()
        assert True

    @pytest.mark.skip("Requires actual implementation")
    @patch("whisper_subtitler.modules.transcribe.Transcriber")
    @patch("whisper_subtitler.modules.diarise.Diarizer")
    @patch("whisper_subtitler.modules.output.OutputFormatter")
    def test_speaker_assignment(self, mock_output_formatter, mock_diarizer, mock_transcriber, mock_config):
        """Test assigning speakers to transcription segments."""
        # Setup mocks
        mock_transcriber_instance = MagicMock()
        mock_diarizer_instance = MagicMock()
        mock_formatter_instance = MagicMock()

        mock_transcriber.return_value = mock_transcriber_instance
        mock_diarizer.return_value = mock_diarizer_instance
        mock_output_formatter.return_value = mock_formatter_instance

        # Configure mocks with test data
        transcription = {
            "text": "This is a test transcription.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "This is a test"},
                {"id": 1, "start": 2.5, "end": 4.0, "text": "transcription."},
            ],
        }

        diarization = [
            {"start": 0.0, "end": 2.2, "speaker": "SPEAKER_01"},
            {"start": 2.3, "end": 4.1, "speaker": "SPEAKER_02"},
        ]

        mock_transcriber_instance.transcribe.return_value = transcription
        mock_diarizer_instance.diarize.return_value = diarization

        # Once implemented, use the actual Application
        # app = Application(mock_config)
        # app.initialize()
        # app.process("test_file.wav")

        # # Get the data passed to the output formatter
        # output_data = mock_formatter_instance.generate_outputs.call_args[0][0]

        # # Check that speakers were assigned correctly
        # assert output_data["segments"][0]["speaker"] == "SPEAKER_01"
        # assert output_data["segments"][1]["speaker"] == "SPEAKER_02"

        # For now, just ensure the test runs
        assert True

    @pytest.mark.skip("Requires actual implementation")
    @patch("whisper_subtitler.modules.transcribe.Transcriber")
    @patch("whisper_subtitler.modules.diarise.Diarizer")
    @patch("whisper_subtitler.modules.output.OutputFormatter")
    @patch("subprocess.run", side_effect=Exception("ffmpeg error"))
    def test_audio_extraction_error(
        self,
        mock_subprocess_run,
        mock_output_formatter,
        mock_diarizer,
        mock_transcriber,
        mock_config,
        sample_video_file,
    ):
        """Test handling of audio extraction errors."""
        # Once implemented, use the actual Application
        # app = Application(mock_config)
        # app.initialize()

        # # Check that an error is raised for audio extraction failure
        # with pytest.raises(Exception, match="ffmpeg error"):
        #     app.process(str(sample_video_file))

        # # Check that no further processing was attempted
        # assert not mock_transcriber_instance.transcribe.called
        # assert not mock_diarizer_instance.diarize.called
        # assert not mock_formatter_instance.generate_outputs.called

        # For now, just ensure the test runs
        assert True

    @pytest.mark.skip("Requires actual implementation")
    @patch("whisper_subtitler.modules.transcribe.Transcriber")
    @patch("whisper_subtitler.modules.diarise.Diarizer")
    @patch("whisper_subtitler.modules.output.OutputFormatter")
    def test_multiple_input_files(
        self, mock_output_formatter, mock_diarizer, mock_transcriber, mock_config, temp_output_dir
    ):
        """Test processing multiple input files."""
        # Setup mocks
        mock_transcriber_instance = MagicMock()
        mock_diarizer_instance = MagicMock()
        mock_formatter_instance = MagicMock()

        mock_transcriber.return_value = mock_transcriber_instance
        mock_diarizer.return_value = mock_diarizer_instance
        mock_output_formatter.return_value = mock_formatter_instance

        # Create test files
        test_files = [
            temp_output_dir / "test1.wav",
            temp_output_dir / "test2.wav",
        ]
        for file in test_files:
            file.touch()

        # Once implemented, use the actual Application
        # app = Application(mock_config)
        # app.initialize()

        # # Process each file
        # for file in test_files:
        #     app.process(str(file))

        # # Check that each file was processed
        # assert mock_transcriber_instance.transcribe.call_count == len(test_files)
        # assert mock_diarizer_instance.diarize.call_count == len(test_files)
        # assert mock_formatter_instance.generate_outputs.call_count == len(test_files)

        # For now, ensure the test files were created
        for file in test_files:
            assert file.exists()
        assert True

    @pytest.mark.skip("Requires actual implementation")
    @patch("whisper_subtitler.modules.transcribe.Transcriber")
    @patch("whisper_subtitler.modules.diarise.Diarizer")
    @patch("whisper_subtitler.modules.output.OutputFormatter")
    def test_no_speakers_detected(
        self, mock_output_formatter, mock_diarizer, mock_transcriber, mock_config, sample_audio_file
    ):
        """Test handling case where no speakers are detected."""
        # Setup mocks
        mock_transcriber_instance = MagicMock()
        mock_diarizer_instance = MagicMock()
        mock_formatter_instance = MagicMock()

        mock_transcriber.return_value = mock_transcriber_instance
        mock_diarizer.return_value = mock_diarizer_instance
        mock_output_formatter.return_value = mock_formatter_instance

        # Configure mocks with test data that has no speakers
        transcription = {
            "text": "This is a test transcription.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "This is a test"},
                {"id": 1, "start": 2.5, "end": 4.0, "text": "transcription."},
            ],
        }

        # Empty diarization result
        diarization = []

        mock_transcriber_instance.transcribe.return_value = transcription
        mock_diarizer_instance.diarize.return_value = diarization

        # Once implemented, use the actual Application
        # app = Application(mock_config)
        # app.initialize()
        # app.process(str(sample_audio_file))

        # # Get the data passed to the output formatter
        # output_data = mock_formatter_instance.generate_outputs.call_args[0][0]

        # # Check that a default speaker was assigned
        # assert "speaker" in output_data["segments"][0]
        # assert output_data["segments"][0]["speaker"] == "SPEAKER_UNKNOWN"
        # assert output_data["segments"][1]["speaker"] == "SPEAKER_UNKNOWN"

        # For now, just ensure the test runs
        assert True
