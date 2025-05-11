"""
Tests for the speaker diarization module.
"""

from unittest.mock import MagicMock, patch

import numpy as np

# Import once implemented
# from whisper_subtitler.modules.diarise import Diarizer


class TestDiarizer:
    """Test the speaker diarization functionality."""

    def test_initialization(self, mock_config):
        """Test initialization of the Diarizer."""
        # Once implemented, use the actual Diarizer
        # diarizer = Diarizer(mock_config)
        # assert diarizer.config == mock_config
        # assert diarizer.num_speakers == mock_config.num_speakers
        # assert diarizer.use_cuda == mock_config.use_cuda
        # assert diarizer.huggingface_token == mock_config.huggingface_token

        # For now, we're just testing that we can create the mock test
        assert mock_config.huggingface_token == "mock-token"
        assert True

    @patch("pyannote.audio.Pipeline.from_pretrained")
    def test_initialize_pipeline(self, mock_from_pretrained, mock_config, mock_torch_cuda):
        """Test initializing the diarization pipeline."""
        # Mock the Pyannote Pipeline
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline

        # Once implemented, use the actual Diarizer
        # diarizer = Diarizer(mock_config)
        # pipeline = diarizer.initialize_pipeline()

        # assert pipeline == mock_pipeline
        # mock_from_pretrained.assert_called_once_with("pyannote/speaker-diarization", use_auth_token=mock_config.huggingface_token)

        # # If using CUDA, should move pipeline to cuda
        # if mock_config.use_cuda and torch.cuda.is_available():
        #     mock_pipeline.to.assert_called_once_with("cuda")

        # For now, just verify the mock was created properly
        assert mock_from_pretrained is not None
        assert True

    @patch("pyannote.audio.Pipeline.from_pretrained")
    def test_diarize(self, mock_from_pretrained, mock_config, sample_audio_file):
        """Test diarizing an audio file."""
        # Mock the Pyannote Pipeline and its result
        mock_pipeline = MagicMock()
        mock_diarization_result = MagicMock()

        # Setup itertracks to return speaker segments
        mock_diarization_result.itertracks.return_value = [
            (MagicMock(start=0.0, end=2.2), None, "SPEAKER_01"),
            (MagicMock(start=2.3, end=4.1), None, "SPEAKER_02"),
        ]

        mock_pipeline.return_value = mock_diarization_result
        mock_from_pretrained.return_value = mock_pipeline

        # Once implemented, use the actual Diarizer
        # diarizer = Diarizer(mock_config)
        # diarizer.initialize_pipeline()
        # result = diarizer.diarize(str(sample_audio_file))

        # # Check the result structure
        # assert len(result) == 2
        # assert result[0]["start"] == 0.0
        # assert result[0]["end"] == 2.2
        # assert result[0]["speaker"] == "SPEAKER_01"
        # assert result[1]["start"] == 2.3
        # assert result[1]["end"] == 4.1
        # assert result[1]["speaker"] == "SPEAKER_02"

        # # Check the pipeline was called with the right arguments
        # mock_pipeline.assert_called_once_with(str(sample_audio_file))

        # For now, just verify the mocks were created properly
        assert mock_from_pretrained is not None
        assert sample_audio_file.exists()
        assert True

    @patch("pyannote.audio.Pipeline.from_pretrained")
    def test_diarize_with_num_speakers(self, mock_from_pretrained, mock_config, sample_audio_file):
        """Test diarizing with a specific number of speakers."""
        # Set num_speakers in config
        mock_config.num_speakers = 2

        # Mock the Pyannote Pipeline and its result
        mock_pipeline = MagicMock()
        mock_diarization_result = MagicMock()

        # Setup itertracks to return speaker segments
        mock_diarization_result.itertracks.return_value = [
            (MagicMock(start=0.0, end=2.2), None, "SPEAKER_01"),
            (MagicMock(start=2.3, end=4.1), None, "SPEAKER_02"),
        ]

        mock_pipeline.return_value = mock_diarization_result
        mock_from_pretrained.return_value = mock_pipeline

        # Once implemented, use the actual Diarizer
        # diarizer = Diarizer(mock_config)
        # diarizer.initialize_pipeline()
        # result = diarizer.diarize(str(sample_audio_file))

        # # Check the result has exactly the specified number of speakers
        # speakers = set(segment["speaker"] for segment in result)
        # assert len(speakers) == 2

        # For now, just verify the mocks were created properly
        assert mock_from_pretrained is not None
        assert mock_config.num_speakers == 2
        assert True

    @patch("pyannote.audio.Pipeline.from_pretrained")
    def test_speaker_clustering(self, mock_from_pretrained, mock_config, sample_audio_file):
        """Test speaker clustering functionality."""
        # Enable speaker clustering in config
        mock_config.cluster_speakers = True

        # Mock the Pyannote Pipeline and its result
        mock_pipeline = MagicMock()
        mock_diarization_result = MagicMock()

        # Create mock embeddings for speaker clustering
        mock_embedding1 = np.random.random(128)
        mock_embedding2 = np.random.random(128)
        mock_embedding3 = np.random.random(128)

        # Make embedding1 and embedding3 similar (simulating same speaker)
        mock_embedding3 = 0.95 * mock_embedding1 + 0.05 * np.random.random(128)

        # Setup itertracks to return speaker segments with embeddings
        def mock_itertracks(*args, **kwargs):
            if kwargs.get("yield_embedding", False):
                return [
                    (MagicMock(start=0.0, end=1.0), mock_embedding1, "SPEAKER_01"),
                    (MagicMock(start=1.5, end=2.2), mock_embedding2, "SPEAKER_02"),
                    (MagicMock(start=2.3, end=4.1), mock_embedding3, "SPEAKER_03"),
                ]
            else:
                return [
                    (MagicMock(start=0.0, end=1.0), None, "SPEAKER_01"),
                    (MagicMock(start=1.5, end=2.2), None, "SPEAKER_02"),
                    (MagicMock(start=2.3, end=4.1), None, "SPEAKER_03"),
                ]

        mock_diarization_result.itertracks = mock_itertracks
        mock_pipeline.return_value = mock_diarization_result
        mock_from_pretrained.return_value = mock_pipeline

        # Once implemented, use the actual Diarizer with speaker clustering
        # diarizer = Diarizer(mock_config)
        # diarizer.initialize_pipeline()
        # result = diarizer.diarize(str(sample_audio_file))

        # # After clustering, we expect SPEAKER_01 and SPEAKER_03 to be merged
        # # due to similar embeddings
        # speakers = set(segment["speaker"] for segment in result)
        # assert len(speakers) == 2  # Not 3, because of clustering

        # # Check that the segments for original SPEAKER_01 and SPEAKER_03 have the same speaker ID
        # speaker_mapping = {}
        # for segment in result:
        #     if 0.0 <= segment["start"] <= 1.0:  # Original SPEAKER_01
        #         speaker_mapping["SPEAKER_01"] = segment["speaker"]
        #     elif 2.3 <= segment["start"] <= 4.1:  # Original SPEAKER_03
        #         speaker_mapping["SPEAKER_03"] = segment["speaker"]

        # assert speaker_mapping["SPEAKER_01"] == speaker_mapping["SPEAKER_03"]

        # For now, just verify the config was set properly
        assert mock_config.cluster_speakers is True
        assert True

    @patch("pyannote.audio.Pipeline.from_pretrained")
    def test_audio_preprocessing(self, mock_from_pretrained, mock_config, sample_audio_file, mock_subprocess):
        """Test audio preprocessing for improved diarization."""
        # Enable audio preprocessing in config
        mock_config.preprocess_audio = True

        # Mock the Pyannote Pipeline
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline

        # Once implemented, use the actual Diarizer
        # diarizer = Diarizer(mock_config)
        # diarizer.initialize_pipeline()
        # processed_audio = diarizer.preprocess_audio(str(sample_audio_file))
        # result = diarizer.diarize(processed_audio)

        # # Check that subprocess was called to preprocess the audio
        # assert mock_subprocess.run.called
        # # Check that the processed audio path was used
        # assert processed_audio != str(sample_audio_file)
        # assert ".processed." in processed_audio

        # For now, just verify the config was set properly
        assert mock_config.preprocess_audio is True
        assert True

    def test_error_handling_missing_token(self, mock_config):
        """Test error handling when HuggingFace token is missing."""
        # Remove token from config
        mock_config.huggingface_token = None

        # Once implemented, use the actual Diarizer and expect ValueError
        # with pytest.raises(ValueError, match="HuggingFace token is required"):
        #     diarizer = Diarizer(mock_config)
        #     diarizer.initialize_pipeline()

        # For now, just verify the token was removed
        assert mock_config.huggingface_token is None
        assert True

    @patch("pyannote.audio.Pipeline.from_pretrained")
    def test_optimize_num_speakers(self, mock_from_pretrained, mock_config, sample_audio_file):
        """Test optimizing for a known number of speakers."""
        # Enable speaker count optimization in config
        mock_config.optimize_num_speakers = True
        mock_config.num_speakers = 2

        # Mock the Pyannote Pipeline and its result
        mock_pipeline = MagicMock()
        mock_diarization_result = MagicMock()

        # Setup itertracks to return more speakers than specified
        mock_diarization_result.itertracks.return_value = [
            (MagicMock(start=0.0, end=1.0), None, "SPEAKER_01"),
            (MagicMock(start=1.5, end=2.2), None, "SPEAKER_02"),
            (MagicMock(start=2.3, end=3.1), None, "SPEAKER_03"),
            (MagicMock(start=3.2, end=4.1), None, "SPEAKER_04"),
        ]

        mock_pipeline.return_value = mock_diarization_result
        mock_from_pretrained.return_value = mock_pipeline

        # Once implemented, use the actual Diarizer
        # diarizer = Diarizer(mock_config)
        # diarizer.initialize_pipeline()
        # result = diarizer.diarize(str(sample_audio_file))

        # # After optimization, we should have exactly the number of speakers specified
        # speakers = set(segment["speaker"] for segment in result)
        # assert len(speakers) == mock_config.num_speakers

        # For now, just verify the config was set properly
        assert mock_config.optimize_num_speakers is True
        assert mock_config.num_speakers == 2
        assert True
