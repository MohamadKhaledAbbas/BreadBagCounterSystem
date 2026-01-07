"""
Unit tests for ROI candidate saver module.

Tests the ROICandidateSaver class and related functions.
"""

import pytest
import os
import json
import tempfile
import shutil
import numpy as np


class TestROICandidateSaverConfig:
    """Tests for ROICandidateSaverConfig class."""
    
    def test_default_config_disabled(self):
        """Test that default config is disabled."""
        from src.classifier.roi_candidate_saver import ROICandidateSaverConfig
        
        config = ROICandidateSaverConfig()
        
        # Should be disabled by default
        assert not config.enabled
    
    def test_config_from_env_vars(self):
        """Test that config reads from environment variables."""
        # This test modifies env vars, so we save and restore them
        import os
        
        original_enabled = os.environ.get('SAVE_ROI_CANDIDATES')
        original_dir = os.environ.get('ROI_CANDIDATES_DIR')
        
        try:
            # Set test values
            os.environ['SAVE_ROI_CANDIDATES'] = 'true'
            os.environ['ROI_CANDIDATES_DIR'] = '/tmp/test_rois'
            
            from src.classifier.roi_candidate_saver import ROICandidateSaverConfig
            
            config = ROICandidateSaverConfig()
            
            assert config.enabled
            assert config.output_dir == '/tmp/test_rois'
            
        finally:
            # Restore original values
            if original_enabled is not None:
                os.environ['SAVE_ROI_CANDIDATES'] = original_enabled
            elif 'SAVE_ROI_CANDIDATES' in os.environ:
                del os.environ['SAVE_ROI_CANDIDATES']
            
            if original_dir is not None:
                os.environ['ROI_CANDIDATES_DIR'] = original_dir
            elif 'ROI_CANDIDATES_DIR' in os.environ:
                del os.environ['ROI_CANDIDATES_DIR']


class TestROICandidateSaver:
    """Tests for ROICandidateSaver class."""
    
    def test_saver_disabled_returns_none(self):
        """Test that disabled saver returns None."""
        from src.classifier.roi_candidate_saver import ROICandidateSaver, ROICandidateSaverConfig
        
        config = ROICandidateSaverConfig()
        config.enabled = False
        
        saver = ROICandidateSaver(config)
        
        result = saver.save_track_candidates(
            track_id=123,
            classification='TestClass',
            confidence=0.9,
            roi_candidates=[]
        )
        
        assert result is None
    
    def test_should_save_track_respects_config(self):
        """Test that _should_save_track respects config flags."""
        from src.classifier.roi_candidate_saver import ROICandidateSaver, ROICandidateSaverConfig
        
        config = ROICandidateSaverConfig()
        config.enabled = True
        config.save_rejected_tracks = False
        config.save_uncertain_tracks = False
        
        saver = ROICandidateSaver(config)
        
        # Should save normal classifications
        assert saver._should_save_track('Brown_Orange_Small')
        
        # Should not save rejected when disabled
        assert not saver._should_save_track('Rejected')
        
        # Should not save uncertain when disabled
        assert not saver._should_save_track('Uncertain')
        assert not saver._should_save_track('Unknown')
    
    def test_get_class_directory_creates_dir(self):
        """Test that _get_class_directory creates the directory."""
        from src.classifier.roi_candidate_saver import ROICandidateSaver, ROICandidateSaverConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ROICandidateSaverConfig()
            config.enabled = True
            config.output_dir = tmpdir
            
            saver = ROICandidateSaver(config)
            
            class_dir = saver._get_class_directory('TestClass')
            
            assert os.path.exists(class_dir)
            assert os.path.isdir(class_dir)
            assert class_dir == os.path.join(tmpdir, 'TestClass')
    
    def test_save_track_candidates_saves_metadata(self):
        """Test that save_track_candidates saves metadata JSON."""
        from src.classifier.roi_candidate_saver import ROICandidateSaver, ROICandidateSaverConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ROICandidateSaverConfig()
            config.enabled = True
            config.output_dir = tmpdir
            
            saver = ROICandidateSaver(config)
            
            # Create a test ROI (simple 10x10 BGR image)
            test_roi = np.zeros((10, 10, 3), dtype=np.uint8)
            test_roi[:] = [100, 150, 200]  # BGR color
            
            roi_candidates = [
                {
                    'roi': test_roi,
                    'sharpness': 500.0,
                    'quality': 0.85,
                    'size': (10, 10),
                    'frame_index': 100,
                    'confidence': 0.9,
                    'state': 'closed',
                    'bbox': (50.0, 50.0, 60.0, 60.0)
                }
            ]
            
            metadata_path = saver.save_track_candidates(
                track_id=12345,
                classification='TestClass',
                confidence=0.9,
                roi_candidates=roi_candidates
            )
            
            # Check metadata file exists
            assert metadata_path is not None
            assert os.path.exists(metadata_path)
            
            # Check metadata content
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            assert metadata['track_id'] == 12345
            assert metadata['final_classification'] == 'TestClass'
            assert metadata['confidence'] == 0.9
            assert metadata['total_roi_count'] == 1
            assert len(metadata['roi_candidates']) == 1
            
            # Check ROI metadata
            roi_meta = metadata['roi_candidates'][0]
            assert roi_meta['roi_index'] == 0
            assert roi_meta['sharpness'] == 500.0
            assert roi_meta['quality'] == 0.85
            assert roi_meta['state'] == 'closed'
    
    def test_save_track_candidates_saves_images(self):
        """Test that save_track_candidates saves ROI images."""
        from src.classifier.roi_candidate_saver import ROICandidateSaver, ROICandidateSaverConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ROICandidateSaverConfig()
            config.enabled = True
            config.output_dir = tmpdir
            
            saver = ROICandidateSaver(config)
            
            # Create a test ROI (simple 50x50 BGR image)
            test_roi = np.zeros((50, 50, 3), dtype=np.uint8)
            test_roi[:] = [100, 150, 200]  # BGR color
            
            roi_candidates = [
                {
                    'roi': test_roi,
                    'sharpness': 500.0,
                    'quality': 0.85,
                    'size': (50, 50),
                    'frame_index': 100,
                    'confidence': 0.9,
                    'state': 'closed',
                    'bbox': (50.0, 50.0, 100.0, 100.0)
                }
            ]
            
            saver.save_track_candidates(
                track_id=12345,
                classification='TestClass',
                confidence=0.9,
                roi_candidates=roi_candidates
            )
            
            # Check that image file was created
            class_dir = os.path.join(tmpdir, 'TestClass')
            jpg_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
            
            assert len(jpg_files) == 1
            assert 'track_12345' in jpg_files[0]
            assert 'quality_0.85' in jpg_files[0]
    
    def test_get_stats_returns_correct_counts(self):
        """Test that get_stats returns correct statistics."""
        from src.classifier.roi_candidate_saver import ROICandidateSaver, ROICandidateSaverConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ROICandidateSaverConfig()
            config.enabled = True
            config.output_dir = tmpdir
            
            saver = ROICandidateSaver(config)
            
            # Create test ROIs
            test_roi = np.zeros((50, 50, 3), dtype=np.uint8)
            
            roi_candidates = [
                {
                    'roi': test_roi,
                    'sharpness': 500.0,
                    'quality': 0.8,
                    'size': (50, 50),
                    'frame_index': 100,
                    'confidence': 0.9,
                    'state': 'closed',
                    'bbox': (50.0, 50.0, 100.0, 100.0)
                }
            ]
            
            # Save to multiple classes
            saver.save_track_candidates(123, 'ClassA', 0.9, roi_candidates)
            saver.save_track_candidates(124, 'ClassA', 0.8, roi_candidates)
            saver.save_track_candidates(125, 'ClassB', 0.7, roi_candidates)
            
            stats = saver.get_stats()
            
            assert stats['enabled']
            assert stats['total_tracks'] == 3
            assert stats['total_rois'] == 3
            assert 'ClassA' in stats['classes']
            assert 'ClassB' in stats['classes']
            assert stats['classes']['ClassA']['tracks'] == 2
            assert stats['classes']['ClassB']['tracks'] == 1


class TestConvenienceFunction:
    """Tests for save_track_roi_candidates convenience function."""
    
    def test_save_track_roi_candidates_uses_singleton(self):
        """Test that convenience function uses global singleton."""
        from src.classifier.roi_candidate_saver import save_track_roi_candidates, get_roi_candidate_saver
        import src.classifier.roi_candidate_saver as saver_module
        
        # Reset singleton
        saver_module._saver_instance = None
        
        # First call creates singleton
        result1 = save_track_roi_candidates(
            track_id=123,
            classification='Test',
            confidence=0.9,
            roi_candidates=[]
        )
        
        # Get singleton
        saver1 = get_roi_candidate_saver()
        saver2 = get_roi_candidate_saver()
        
        # Should be same instance
        assert saver1 is saver2
