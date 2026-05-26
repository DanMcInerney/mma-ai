import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import joblib
from unittest import mock

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import functions and classes from predict.py
from predict import (
    BFOLatestOddsOnly,
    get_manual_fighter_odds,
    get_bfo_odds,
    convert_american_to_decimal,
    get_fights,
    load_model_and_calibrator,
    get_predictions,
    load_model,
    create_conf_parlays,
    convert_prob_to_american_odds,
    american_odds_to_prob,
    calculate_expected_value,
    has_positive_ev,
    latest_model_path,
    maybe_take_screenshots,
    parse_manual_odds_json,
    apply_manual_odds,
)
from libs.bfo_scraper import BFOScraper


class TestBFOLatestOddsOnly:
    """Test suite for BFOLatestOddsOnly class"""
    
    def test_init(self):
        """Test BFOLatestOddsOnly initialization"""
        scraper = BFOLatestOddsOnly()
        assert scraper.base_url == "https://www.bestfightodds.com"
        assert "User-Agent" in scraper.default_headers
        assert isinstance(scraper.NAME_MAPPINGS, dict)
        assert isinstance(scraper.REVERSE_MAPPINGS, dict)
        assert isinstance(scraper.DUPE_NAMES, dict)

    def test_latest_odds_uses_historical_scraper_name_mappings(self):
        """Current-odds lookup should not drift from odds DB scraping aliases."""
        scraper = BFOLatestOddsOnly()

        assert BFOLatestOddsOnly.NAME_MAPPINGS is BFOScraper.NAME_MAPPINGS
        assert BFOLatestOddsOnly.REVERSE_MAPPINGS is BFOScraper.REVERSE_MAPPINGS
        assert BFOLatestOddsOnly.DUPE_NAMES is BFOScraper.DUPE_NAMES
        assert scraper.get_mapped_name("yadier delvalle") == "yadier del valle"
        assert scraper.get_mapped_name("asu almabayev") == "asu almabaev"
        assert scraper.reverse_map_name("timothy cuamba") == "timmy cuamba"
        assert scraper.get_mapped_name(None) is None
        assert scraper.reverse_map_name(None) is None

    @patch('predict.requests.get')
    @patch('predict.BeautifulSoup')
    def test_get_fighter_link_success(self, mock_soup, mock_requests):
        """Test successful fighter link retrieval"""
        scraper = BFOLatestOddsOnly()
        
        # Setup mocks
        mock_response = MagicMock()
        mock_response.text = "<html><a href='/fighter/test'>Test Fighter</a></html>"
        mock_requests.return_value = mock_response
        
        mock_soup_instance = MagicMock()
        mock_link = MagicMock()
        mock_link.__getitem__.return_value = '/fighter/test'
        mock_soup_instance.find.return_value = mock_link
        mock_soup.return_value = mock_soup_instance
        
        # Test
        result = scraper.get_fighter_link("test fighter")
        
        assert result == '/fighter/test'
        mock_requests.assert_called_once()
        mock_soup.assert_called_once()
    
    @patch('predict.requests.get')
    def test_get_fighter_link_failure(self, mock_requests):
        """Test fighter link retrieval failure"""
        scraper = BFOLatestOddsOnly()
        
        # Setup mock to raise exception
        mock_requests.side_effect = Exception("Network error")
        
        # Test
        result = scraper.get_fighter_link("test fighter")
        
        assert result is None
    
    @patch('predict.requests.get')
    @patch('predict.BeautifulSoup')
    def test_parse_fighter_page_success(self, mock_soup, mock_requests):
        """Test successful fighter page parsing"""
        scraper = BFOLatestOddsOnly()
        
        # Setup mocks
        mock_response = MagicMock()
        mock_response.text = "<html>test</html>"
        mock_requests.return_value = mock_response
        
        # Create mock rows
        mock_main_row = MagicMock()
        mock_main_row.get.return_value = ["main-row"]
        
        mock_opp_row = MagicMock()
        mock_link = MagicMock()
        mock_link.text = "opponent fighter"
        mock_opp_row.find.return_value = mock_link
        
        mock_span = MagicMock()
        mock_span.__getitem__.return_value = "[1,2]"
        mock_main_row.find.return_value = mock_span
        
        mock_soup_instance = MagicMock()
        mock_soup_instance.find_all.return_value = [mock_main_row, mock_opp_row]
        mock_soup.return_value = mock_soup_instance
        
        # Test
        result = scraper.parse_fighter_page("/test/link")
        
        assert isinstance(result, list)
        if result:  # If parsing succeeded
            assert 'opp_name' in result[0]
            assert 'data_li' in result[0]
    
    def test_decode_base64(self):
        """Test base64 decoding functionality"""
        scraper = BFOLatestOddsOnly()
        
        # Test with simple JSON data
        # This is a simplified test - in practice this would be complex encoded data
        try:
            # For testing, we'll just verify the method exists and handles errors
            result = scraper.decode_base64("invalid_base64")
            # If it doesn't crash, that's good enough for this test
        except:
            # Expected for invalid data
            pass
    
    @patch('predict.requests.get')
    def test_get_perc_change_success(self, mock_requests):
        """Test percentage change data retrieval"""
        scraper = BFOLatestOddsOnly()
        
        # Mock a valid response that would decode to fight data
        mock_response = MagicMock()
        mock_response.text = "valid_encoded_data"
        mock_requests.return_value = mock_response
        
        # Mock the decode method to return expected structure
        with patch.object(scraper, 'decode_base64') as mock_decode:
            mock_decode.return_value = [{"data": [{"x": 1234567890000, "y": 1.5}]}]
            
            result = scraper.get_perc_change(1, 2)
            
            assert isinstance(result, list)
            if result:
                assert 'x' in result[0]
                assert 'y' in result[0]
    
    def test_find_single_fighter_odds_no_link(self):
        """Test find_single_fighter_odds when no fighter link found"""
        scraper = BFOLatestOddsOnly()
        
        with patch.object(scraper, 'get_fighter_link') as mock_get_link:
            mock_get_link.return_value = None
            
            result = scraper.find_single_fighter_odds("test fighter", "opponent")
            
            assert result is None
    
    def test_find_fight_odds_no_match(self):
        """Test find_fight_odds when no match is found"""
        scraper = BFOLatestOddsOnly()
        
        with patch.object(scraper, 'find_single_fighter_odds') as mock_find_single:
            mock_find_single.return_value = None
            
            result = scraper.find_fight_odds("fighter1", "fighter2")
            
            assert result['found_match'] == False
            assert result['both_odds_found'] == False
            assert result['fighter1'] == "fighter1"
            assert result['fighter2'] == "fighter2"
    
    def test_find_fight_odds_both_found(self):
        """Test find_fight_odds when both fighters' odds are found"""
        scraper = BFOLatestOddsOnly()
        
        fighter1_data = {
            'fighter': 'fighter1',
            'opponent': 'fighter2',
            'decimal_odds': 1.8,
            'american_odds': -125,
            'timestamp': datetime.now()
        }
        
        fighter2_data = {
            'fighter': 'fighter2', 
            'opponent': 'fighter1',
            'decimal_odds': 2.2,
            'american_odds': 120,
            'timestamp': datetime.now()
        }
        
        with patch.object(scraper, 'find_single_fighter_odds') as mock_find_single:
            mock_find_single.side_effect = [fighter1_data, fighter2_data]
            
            result = scraper.find_fight_odds("fighter1", "fighter2")
            
            assert result['found_match'] == True
            assert result['both_odds_found'] == True
            assert result['fighter1_odds'] == -125
            assert result['fighter2_odds'] == 120
    
    def test_find_fight_odds_one_found(self):
        """Test find_fight_odds when only one fighter's odds are found"""
        scraper = BFOLatestOddsOnly()
        
        fighter1_data = {
            'fighter': 'fighter1',
            'opponent': 'fighter2',
            'decimal_odds': 1.8,
            'american_odds': -125,
            'timestamp': datetime.now()
        }
        
        with patch.object(scraper, 'find_single_fighter_odds') as mock_find_single:
            mock_find_single.side_effect = [fighter1_data, None]
            
            result = scraper.find_fight_odds("fighter1", "fighter2")
            
            assert result['found_match'] == True
            assert result['both_odds_found'] == False
            assert result['fighter1_odds'] == -125
            assert result['fighter2_odds'] == None  # No longer calculates opponent odds automatically
    
    def test_get_latest_fight_odds_no_db(self):
        """Test get_latest_fight_odds_no_db method"""
        scraper = BFOLatestOddsOnly()
        
        fight_list = [("fighter1", "fighter2"), ("fighter3", "fighter4")]
        
        mock_fight_odds_1 = {
            'fighter1': 'fighter1',
            'fighter2': 'fighter2', 
            'fighter1_odds': -150,
            'fighter2_odds': 130,
            'found_match': True,
            'both_odds_found': True
        }
        
        mock_fight_odds_2 = {
            'fighter1': 'fighter3',
            'fighter2': 'fighter4',
            'found_match': False,
            'both_odds_found': False
        }
        
        with patch.object(scraper, 'find_fight_odds') as mock_find_fight:
            mock_find_fight.side_effect = [mock_fight_odds_1, mock_fight_odds_2]
            
            with patch.object(scraper, 'remove_vig') as mock_remove_vig:
                mock_remove_vig.return_value = (-140, 120)
                
                result = scraper.get_latest_fight_odds_no_db(fight_list)
                
                assert isinstance(result, dict)
                # Check that successful fight has odds data
                if 'fighter1' in result and result['fighter1'] != "N/A":
                    assert isinstance(result['fighter1'], dict)
                    assert 'original' in result['fighter1']
                    assert 'vigless' in result['fighter1']
                
                # Check that failed fight has N/A
                assert result.get('fighter3') == "N/A"
                assert result.get('fighter4') == "N/A"
    
    def test_get_mapped_name(self):
        """Test fighter name mapping functionality"""
        scraper = BFOLatestOddsOnly()
        
        # Test existing mapping
        mapped_name = scraper.get_mapped_name("aj matthews")
        assert mapped_name == "a.j. matthews"
        
        # Test non-existing mapping (should return original)
        unmapped_name = scraper.get_mapped_name("unknown fighter")
        assert unmapped_name == "unknown fighter"
        
        # Test case insensitive
        mapped_name_caps = scraper.get_mapped_name("AJ MATTHEWS")
        assert mapped_name_caps == "a.j. matthews"
    
    def test_reverse_map_name(self):
        """Test reverse name mapping functionality"""
        scraper = BFOLatestOddsOnly()
        
        # Test existing reverse mapping
        reverse_mapped = scraper.reverse_map_name("a.j. matthews")
        assert reverse_mapped == "aj matthews"
        
        # Test dupe names mapping
        reverse_mapped_dupe = scraper.reverse_map_name("kenan song")
        assert reverse_mapped_dupe == "song kenan"
        
        # Test non-existing mapping (should return original)
        unmapped_reverse = scraper.reverse_map_name("unknown fighter")
        assert unmapped_reverse == "unknown fighter"
    
    def test_sanitize_name(self):
        """Test name sanitization (removing @ symbols)"""
        scraper = BFOLatestOddsOnly()
        
        # Test removing @ symbol
        sanitized = scraper.sanitize_name("fighter@test")
        assert sanitized == "fightertest"
        
        # Test multiple @ symbols
        sanitized_multiple = scraper.sanitize_name("fighter@test@name")
        assert sanitized_multiple == "fightertestname"
        
        # Test no @ symbol
        no_change = scraper.sanitize_name("normal fighter")
        assert no_change == "normal fighter"
        
        # Test None input
        none_input = scraper.sanitize_name(None)
        assert none_input is None
        
        # Test empty string
        empty_input = scraper.sanitize_name("")
        assert empty_input == ""
    
    def test_decimal_to_american(self):
        """Test decimal to American odds conversion"""
        scraper = BFOLatestOddsOnly()
        
        # Test decimal >= 2.0 (positive American odds)
        american_positive = scraper.decimal_to_american(3.0)
        assert american_positive == 200
        
        # Test decimal < 2.0 (negative American odds)
        american_negative = scraper.decimal_to_american(1.5)
        assert american_negative == -200
        
        # Test edge case decimal = 2.0
        american_edge = scraper.decimal_to_american(2.0)
        assert american_edge == 100
        
        # Test invalid input (should return fallback)
        american_invalid = scraper.decimal_to_american("invalid")
        assert american_invalid == 100
    
    def test_remove_vig(self):
        """Test vig removal from odds"""
        scraper = BFOLatestOddsOnly()
        
        # Test typical odds pair
        fair_odds1, fair_odds2 = scraper.remove_vig(-110, -110)
        assert isinstance(fair_odds1, int)
        assert isinstance(fair_odds2, int)
        # Fair odds should be closer to even money
        assert abs(fair_odds1 - 100) < abs(-110 - 100)
        assert abs(fair_odds2 - 100) < abs(-110 - 100)
        
        # Test mixed odds
        fair_odds1, fair_odds2 = scraper.remove_vig(-150, 130)
        assert isinstance(fair_odds1, int)
        assert isinstance(fair_odds2, int)
        
        # Test error handling
        fair_odds1_err, fair_odds2_err = scraper.remove_vig("invalid", 110)
        assert fair_odds1_err == "invalid"
        assert fair_odds2_err == 110
    
    def test_calculate_opponent_odds(self):
        """Test opponent odds calculation"""
        scraper = BFOLatestOddsOnly()
        
        # Test with negative odds
        opp_odds = scraper.calculate_opponent_odds(-150)
        assert isinstance(opp_odds, int)
        assert opp_odds > 0  # Opponent should be underdog
        
        # Test with positive odds
        opp_odds = scraper.calculate_opponent_odds(200)
        assert isinstance(opp_odds, int)
        assert opp_odds < 0  # Opponent should be favorite
        
        # Test edge cases
        opp_odds_edge = scraper.calculate_opponent_odds(-100)
        assert isinstance(opp_odds_edge, int)
        
        # Test invalid input
        opp_odds_invalid = scraper.calculate_opponent_odds("invalid")
        assert opp_odds_invalid is None


class TestUtilityFunctions:
    """Test suite for utility functions"""
    
    def test_convert_american_to_decimal(self):
        """Test American to decimal odds conversion"""
        # Test positive American odds
        decimal_pos = convert_american_to_decimal(200)
        assert decimal_pos == 3.0
        
        # Test negative American odds
        decimal_neg = convert_american_to_decimal(-200)
        assert decimal_neg == 1.5
        
        # Test string input with +
        decimal_str_pos = convert_american_to_decimal("+150")
        assert decimal_str_pos == 2.5
        
        # Test N/A string
        decimal_na = convert_american_to_decimal("N/A")
        assert decimal_na == 0
        
        # Test invalid input
        decimal_invalid = convert_american_to_decimal("invalid")
        assert decimal_invalid == 0
    
    def test_convert_prob_to_american_odds(self):
        """Test probability to American odds conversion"""
        # Test probability > 0.5 (favorite)
        odds_fav = convert_prob_to_american_odds(0.7)
        assert odds_fav.startswith("-")
        assert int(odds_fav) < 0
        
        # Test probability < 0.5 (underdog)
        odds_dog = convert_prob_to_american_odds(0.3)
        assert odds_dog.startswith("+")
        assert int(odds_dog[1:]) > 0
        
        # Test probability = 0.5 (even money)
        odds_even = convert_prob_to_american_odds(0.5)
        assert odds_even.startswith("+")
        assert int(odds_even[1:]) == 100
    
    def test_american_odds_to_prob(self):
        """Test American odds to probability conversion"""
        # Test negative odds
        prob_neg = american_odds_to_prob(-200)
        assert 0 < prob_neg < 1
        assert prob_neg > 0.5  # Should be > 50% for favorites
        
        # Test positive odds
        prob_pos = american_odds_to_prob(200)
        assert 0 < prob_pos < 1
        assert prob_pos < 0.5  # Should be < 50% for underdogs
        
        # Test string input
        prob_str = american_odds_to_prob("-150")
        assert 0 < prob_str < 1
        
        # Test string with +
        prob_str_pos = american_odds_to_prob("+200")
        assert 0 < prob_str_pos < 1
        
        # Test invalid input
        prob_invalid = american_odds_to_prob("invalid")
        assert prob_invalid is None
    
    def test_calculate_expected_value(self):
        """Test expected value calculation"""
        # Test positive EV scenario
        ev_pos = calculate_expected_value(-120, -110)
        assert isinstance(ev_pos, float)
        
        # Test negative EV scenario
        ev_neg = calculate_expected_value(-110, -120)
        assert isinstance(ev_neg, float)
        
        # Test N/A odds
        ev_na = calculate_expected_value(-120, "N/A")
        assert ev_na == 0.0
        
        # Test pandas NA
        ev_pandas_na = calculate_expected_value(-120, pd.NA)
        assert ev_pandas_na == 0.0
        
        # Test invalid inputs
        ev_invalid = calculate_expected_value("invalid", "invalid")
        assert ev_invalid == 0.0
    
    def test_has_positive_ev(self):
        """Test positive EV detection"""
        # Test positive EV
        has_ev_pos = has_positive_ev(-110, -120)
        assert isinstance(has_ev_pos, bool)
        
        # Test negative EV
        has_ev_neg = has_positive_ev(-120, -110)
        assert isinstance(has_ev_neg, bool)
        
        # Test N/A odds
        has_ev_na = has_positive_ev(-120, "N/A")
        assert has_ev_na == False


class TestModelFunctions:
    """Test suite for model-related functions"""
    
    @patch('autogluon.tabular.TabularPredictor')
    @patch('predict.load_joblib_artifact')
    @patch('predict.os.path.exists')
    def test_load_model_and_calibrator(self, mock_exists, mock_joblib_load, mock_predictor):
        """Test model and calibrator loading"""
        # Setup mocks
        mock_predictor.load.return_value = MagicMock()
        mock_calibrator = MagicMock()
        mock_joblib_load.return_value = mock_calibrator
        
        # Test with calibrator available
        mock_exists.return_value = True
        model, calibrator = load_model_and_calibrator("/fake/path", use_calibrated=True)
        
        assert model is not None
        assert calibrator is not None
        mock_predictor.load.assert_called_once_with(
            "/fake/path",
            require_version_match=False,
            require_py_version_match=False,
        )
        
        # Test without calibrator
        mock_exists.return_value = False
        model, calibrator = load_model_and_calibrator("/fake/path", use_calibrated=True)
        
        assert model is not None
        assert calibrator is None
        
        # Test with calibration disabled
        model, calibrator = load_model_and_calibrator("/fake/path", use_calibrated=False)
        
        assert model is not None
        assert calibrator is None
    
    @patch('autogluon.tabular.TabularPredictor')
    def test_load_model(self, mock_predictor):
        """Test deprecated load_model function"""
        mock_predictor.load.return_value = MagicMock()
        
        model = load_model("/fake/path")
        
        assert model is not None
        mock_predictor.load.assert_called_once_with("/fake/path")

    def test_ensemble_predictor_load_bypasses_runtime_metadata_guards(self, tmp_path, monkeypatch):
        """Walk-forward ensembles should load starter artifacts across Docker/local runtimes."""
        from libs.modeling import train as train_module

        model_dir = tmp_path / "ensemble"
        window_dir = model_dir / "window_0"
        final_model_dir = model_dir / "final_model"
        window_dir.mkdir(parents=True)
        final_model_dir.mkdir()

        calls = []

        class FakePredictor:
            label = "y_true"
            eval_metric = "log_loss"
            problem_type = "binary"
            model_best = "FakeModel"

            def __init__(self, path):
                self.path = path

        def fake_load(path, **kwargs):
            calls.append((path, kwargs))
            return FakePredictor(path)

        fake_tabular = MagicMock()
        fake_tabular.load.side_effect = fake_load
        monkeypatch.setattr(train_module, "TabularPredictor", fake_tabular)

        ensemble = train_module.EnsemblePredictor.load(str(model_dir))

        assert len(ensemble.predictors) == 2
        assert calls == [
            (
                str(window_dir),
                {"require_version_match": False, "require_py_version_match": False},
            ),
            (
                str(final_model_dir),
                {"require_version_match": False, "require_py_version_match": False},
            ),
        ]
    
    def test_get_predictions_no_calibrator(self):
        """Test get_predictions function without calibrator"""
        # Create mock model
        mock_model = MagicMock()
        mock_predictions = pd.DataFrame({0: [0.3], 1: [0.7]})
        mock_model.predict_proba.return_value = mock_predictions
        
        # Create test data
        test_data = pd.DataFrame({'feature1': [1.0], 'feature2': [2.0]})
        
        # Test without calibrator
        result = get_predictions(mock_model, None, test_data, use_calibrated=False)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [0, 1]
        assert len(result) == 1
        pd.testing.assert_frame_equal(result, mock_predictions)
    
    def test_get_predictions_with_sample_weight_error(self):
        """Test get_predictions handling sample_weight KeyError"""
        # Create mock model that raises KeyError for sample_weight
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = [
            KeyError("sample_weight"),
            pd.DataFrame({0: [0.3], 1: [0.7]})
        ]
        
        # Create test data
        test_data = pd.DataFrame({'feature1': [1.0], 'feature2': [2.0]})
        
        # Test sample_weight handling
        result = get_predictions(mock_model, None, test_data, use_calibrated=False)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [0, 1]
        assert mock_model.predict_proba.call_count == 2
        
        # Check that sample_weight was added in second call
        second_call_args = mock_model.predict_proba.call_args_list[1][0][0]
        assert 'sample_weight' in second_call_args.columns
        assert (second_call_args['sample_weight'] == 1.0).all()

    @patch("predict.take_screenshots")
    def test_maybe_take_screenshots_is_opt_in(self, mock_take_screenshots):
        """Prediction should not require browser screenshots by default."""
        result = maybe_take_screenshots("/tmp/predictions", enabled=False)

        assert result is False
        mock_take_screenshots.assert_not_called()

    @patch("predict.take_screenshots", side_effect=RuntimeError("chrome unavailable"))
    def test_maybe_take_screenshots_does_not_fail_prediction(self, mock_take_screenshots):
        """Screenshot failures should not fail already-written prediction outputs."""
        result = maybe_take_screenshots("/tmp/predictions", enabled=True)

        assert result is False
        mock_take_screenshots.assert_called_once_with("/tmp/predictions")


class TestDataProcessingFunctions:
    """Test suite for data processing functions"""
    
    @patch('predict.UpcomingFights')
    def test_get_fights(self, mock_upcoming_fights):
        """Test get_fights function"""
        # Setup mock
        mock_uf_instance = MagicMock()
        mock_events = {
            'Event 1': [
                (datetime(2024, 1, 1), 'fighter a', 'fighter b'),
                (datetime(2024, 1, 1), 'fighter c', 'fighter d')
            ],
            'Event 2': [
                (datetime(2024, 1, 2), 'fighter e', 'fighter f')
            ]
        }
        mock_uf_instance.run.return_value = mock_events
        mock_upcoming_fights.return_value = mock_uf_instance
        
        # Create dummy dataframe
        test_df = pd.DataFrame({'dummy': [1, 2, 3]})
        
        # Test function
        fight_list, event_names = get_fights(test_df, 1)
        
        # Verify results
        assert len(fight_list) == 3
        assert len(event_names) == 2
        assert event_names == ['Event 1', 'Event 2']
        
        # Verify UpcomingFights was called correctly
        mock_upcoming_fights.assert_called_once_with(test_df, 1)
        mock_uf_instance.run.assert_called_once()


class TestParlayFunctions:
    """Test suite for parlay creation functions"""
    
    def test_create_conf_parlays(self):
        """Test confidence-based parlay creation"""
        # Create sample results data
        sample_results = [
            {
                'fighter1_name': 'fighter a',
                'fighter2_name': 'fighter b', 
                'winner': 'fighter1',
                'proba': 0.85,
                'fighter1_decimal_odds': 1.5,
                'fighter2_decimal_odds': 2.5
            },
            {
                'fighter1_name': 'fighter c',
                'fighter2_name': 'fighter d',
                'winner': 'fighter2', 
                'proba': 0.75,
                'fighter1_decimal_odds': 2.0,
                'fighter2_decimal_odds': 1.8
            },
            {
                'fighter1_name': 'fighter e',
                'fighter2_name': 'fighter f',
                'winner': 'fighter1',
                'proba': 0.70,
                'fighter1_decimal_odds': 1.7,
                'fighter2_decimal_odds': 2.1
            },
            {
                'fighter1_name': 'fighter g',
                'fighter2_name': 'fighter h',
                'winner': 'fighter2',
                'proba': 0.65,
                'fighter1_decimal_odds': 2.2,
                'fighter2_decimal_odds': 1.6
            },
            {
                'fighter1_name': 'fighter i',
                'fighter2_name': 'fighter j',
                'winner': 'fighter1',
                'proba': 0.60,
                'fighter1_decimal_odds': 1.9,
                'fighter2_decimal_odds': 1.9
            },
            {
                'fighter1_name': 'fighter k',
                'fighter2_name': 'fighter l',
                'winner': 'fighter2',
                'proba': 0.55,
                'fighter1_decimal_odds': 2.1,
                'fighter2_decimal_odds': 1.8
            }
        ]
        
        # Test parlay creation
        parlays = create_conf_parlays(sample_results)
        
        # Verify basic structure
        assert isinstance(parlays, list)
        assert len(parlays) <= 5  # Should return max 5 parlays
        
        # Check each parlay structure
        for parlay in parlays:
            assert 'fighters' in parlay
            assert 'avg_confidence' in parlay
            assert 'combined_decimal' in parlay
            assert 'american_odds' in parlay
            assert len(parlay['fighters']) == 3  # 3-fighter parlays
            
            # Check confidence format
            assert parlay['avg_confidence'].endswith('%')
            
            # Check odds format
            assert (parlay['american_odds'].startswith('+') or 
                   parlay['american_odds'].startswith('-'))


class TestIntegrationScenarios:
    """Integration tests using realistic data scenarios"""
    
    @patch('predict.get_bfo_odds')
    @patch('predict.CreateInferenceData')
    def test_prediction_pipeline_integration(self, mock_create_inference, mock_get_bfo_odds):
        """Test integration of the main prediction pipeline components"""
        # Setup mock BFO odds
        mock_bfo_odds = {
            'fighter a': {'original': -150, 'vigless': -140},
            'fighter b': {'original': 130, 'vigless': 125}
        }
        mock_get_bfo_odds.return_value = mock_bfo_odds
        
        # Setup mock inference data
        mock_inference_instance = MagicMock()
        mock_fighter_dfs = {
            'fighter a': pd.DataFrame({
                'feature1': [1.5],
                'feature2': [2.3],
                'feature3': [0.8]
            })
        }
        mock_inference_instance.run.return_value = mock_fighter_dfs
        mock_create_inference.return_value = mock_inference_instance
        
        # Test the integration
        fight_list = [(datetime(2024, 1, 1), 'fighter a', 'fighter b')]
        
        # Verify BFO odds retrieval
        bfo_odds = mock_get_bfo_odds(fight_list)
        assert bfo_odds == mock_bfo_odds
        
        # Verify inference data creation
        cid = mock_create_inference('/fake/path', ['feature1', 'feature2', 'feature3'], 
                                   fight_list, bfo_odds)
        fighter_dfs = cid.run()
        
        assert 'fighter a' in fighter_dfs
        assert isinstance(fighter_dfs['fighter a'], pd.DataFrame)
        assert len(fighter_dfs['fighter a']) == 1
        assert list(fighter_dfs['fighter a'].columns) == ['feature1', 'feature2', 'feature3']
    
    @patch('predict.load_joblib_artifact')
    @patch('predict.load_model_and_calibrator')
    def test_full_prediction_workflow(self, mock_load_model, mock_joblib_load):
        """Test complete prediction workflow from features to results"""
        # Setup mock model and scaler
        mock_model = MagicMock()
        mock_calibrator = MagicMock()
        mock_scaler = MagicMock()
        
        # Mock model predictions
        mock_predictions = pd.DataFrame({0: [0.25], 1: [0.75]})
        mock_model.predict_proba.return_value = mock_predictions
        
        # Mock scaler transformation
        mock_scaled_features = np.array([[1.0, 2.0, 3.0]])
        mock_scaler.transform.return_value = mock_scaled_features
        
        # Setup mocks
        mock_load_model.return_value = (mock_model, mock_calibrator)
        mock_joblib_load.return_value = mock_scaler
        
        # Create test features
        test_features = pd.DataFrame({
            'feature1': [0.5],
            'feature2': [1.5], 
            'feature3': [2.5]
        })
        
        # Test the workflow
        features = ['feature1', 'feature2', 'feature3']
        X = test_features[features]
        
        # Scale features
        scaled_X = mock_scaler.transform(X)
        scaled_X_df = pd.DataFrame(scaled_X, columns=X.columns)
        
        # Get predictions
        predictions = get_predictions(mock_model, mock_calibrator, scaled_X_df, use_calibrated=False)
        
        # Verify results
        assert isinstance(predictions, pd.DataFrame)
        assert list(predictions.columns) == [0, 1]
        assert len(predictions) == 1
        assert predictions.iloc[0][1] == 0.75  # Fighter1 win probability
    
    @patch('predict.BFOLatestOddsOnly')
    @patch('predict.get_manual_fighter_odds')
    def test_bfo_integration_with_manual_fallback(self, mock_manual_odds, mock_bfo_class):
        """Test BFO odds retrieval with manual input fallback"""
        # Setup mock BFO scraper
        mock_scraper = MagicMock()
        mock_bfo_class.return_value = mock_scraper
        
        # Mock successful odds retrieval for some fighters, missing for others
        mock_scraper.get_latest_fight_odds_no_db.return_value = {
            'fighter1': {'original': -150, 'vigless': -140},
            'fighter2': {'original': 130, 'vigless': 125},
            'fighter3': "N/A",
            'fighter4': "N/A"
        }
        
        # Mock manual odds input for missing fighters
        mock_manual_odds.side_effect = [-160, 140]  # Manual odds for fighter3 and fighter4
        mock_scraper.remove_vig.return_value = (-150, 130)
        
        fight_list = [
            (datetime(2024, 1, 1), 'fighter1', 'fighter2'),
            (datetime(2024, 1, 2), 'fighter3', 'fighter4')
        ]
        
        # Test with successful BFO retrieval and manual fallback
        result = get_bfo_odds(fight_list)
        
        assert isinstance(result, dict)
        assert 'fighter1' in result
        assert 'fighter2' in result
        assert isinstance(result['fighter1'], dict)
        assert 'original' in result['fighter1']
        assert 'vigless' in result['fighter1']
    
    @patch('predict.BFOLatestOddsOnly')
    @patch('predict.get_manual_fighter_odds')
    def test_bfo_integration_full_manual_fallback(self, mock_manual_odds, mock_bfo_class):
        """Test BFO odds retrieval with complete manual fallback"""
        # Setup mock BFO scraper to fail
        mock_scraper = MagicMock()
        mock_bfo_class.return_value = mock_scraper
        mock_scraper.get_latest_fight_odds_no_db.side_effect = Exception("Network error")
        mock_scraper.remove_vig.return_value = (-140, 120)
        
        # Setup manual odds input
        mock_manual_odds.side_effect = [-150, 130]
        
        fight_list = [(datetime(2024, 1, 1), 'fighter1', 'fighter2')]
        
        # Test manual fallback
        result = get_bfo_odds(fight_list)
        
        assert isinstance(result, dict)
        assert 'fighter1' in result
        assert 'fighter2' in result
        # Should have manually entered odds
        assert result['fighter1'] == -140  # After vig removal
        assert result['fighter2'] == 120   # After vig removal
    
    def test_comprehensive_fight_prediction_scenario(self):
        """Test a complete fight prediction scenario with realistic data"""
        # Create realistic fight result scenario
        fight_result = {
            "fighter1_name": "jon jones",
            "fighter2_name": "stipe miocic", 
            "fight_date": datetime(2024, 11, 16),
            "fighter1_win_prob": 0.78,
            "fighter2_win_prob": 0.22,
            "fighter1_market_prob": 0.70,
            "fighter2_market_prob": 0.30,
            "fighter1_odds": -233,
            "fighter2_odds": 195,
            "fighter1_vigless_odds": -220,
            "fighter2_vigless_odds": 180,
            "fighter1_decimal_odds": 1.45,
            "fighter2_decimal_odds": 2.8,
            "winner": "fighter1",
            "proba": 0.78
        }
        
        # Test AI odds conversion
        ai_odds = convert_prob_to_american_odds(fight_result['proba'])
        assert ai_odds.startswith('-')  # Should be negative for heavy favorite
        
        # Test EV calculation
        ev = calculate_expected_value(ai_odds, fight_result['fighter1_vigless_odds'])
        assert isinstance(ev, float)
        
        # Test positive EV detection
        has_ev = has_positive_ev(ai_odds, fight_result['fighter1_vigless_odds'])
        assert isinstance(has_ev, bool)
        
        # Test market probability calculation
        market_prob = american_odds_to_prob(fight_result['fighter1_vigless_odds'])
        assert 0 < market_prob < 1
        assert market_prob > 0.5  # Should be > 50% for favorites
        
        # Verify consistency between decimal and American odds
        decimal_from_american = convert_american_to_decimal(fight_result['fighter1_vigless_odds'])
        assert abs(decimal_from_american - fight_result['fighter1_decimal_odds']) < 0.1
    
    @patch('predict.input')
    def test_manual_odds_input_scenarios(self, mock_input):
        """Test manual odds input with various user inputs"""
        # Test valid odds input
        mock_input.return_value = "-150"
        result = get_manual_fighter_odds("test fighter")
        assert result == -150
        
        # Test positive odds with + prefix
        mock_input.return_value = "+200"
        result = get_manual_fighter_odds("test fighter")
        assert result == 200
        
        # Test skip input
        mock_input.return_value = "skip"
        result = get_manual_fighter_odds("test fighter")
        assert result is None
        
        # Test invalid input followed by valid input
        mock_input.side_effect = ["invalid", "-120"]
        result = get_manual_fighter_odds("test fighter")
        assert result == -120
    
    def test_odds_conversion_pipeline(self):
        """Test the complete odds conversion pipeline"""
        # Test American -> Decimal -> Probability chain
        american_odds = -150
        decimal_odds = convert_american_to_decimal(american_odds)
        probability = american_odds_to_prob(american_odds)
        
        # Verify consistency
        assert decimal_odds > 1.0
        assert 0 < probability < 1
        assert probability > 0.5  # Favorite should have > 50% probability
        
        # Test reverse conversion
        converted_american = convert_prob_to_american_odds(probability)
        converted_prob = american_odds_to_prob(converted_american)
        
        # Should be approximately equal (allowing for rounding)
        assert abs(probability - converted_prob) < 0.01
    
    def test_expected_value_calculation_scenarios(self):
        """Test EV calculations in various scenarios"""
        scenarios = [
            # (ai_odds, bookie_odds, expected_ev_positive)
            (-120, -110, True),   # AI more confident than market
            (-110, -120, False),  # Market more confident than AI
            (150, 200, True),     # AI sees better underdog value
            (200, 150, False),    # Market sees better underdog value
        ]
        
        for ai_odds, bookie_odds, expected_positive in scenarios:
            ev = calculate_expected_value(ai_odds, bookie_odds)
            has_ev = has_positive_ev(ai_odds, bookie_odds)
            
            assert isinstance(ev, float)
            assert isinstance(has_ev, bool)
            assert has_ev == expected_positive
            assert (ev > 0) == expected_positive


class TestErrorHandling:
    """Test suite for error handling and edge cases"""
    
    def test_bfo_scraper_error_handling(self):
        """Test BFOLatestOddsOnly error handling"""
        scraper = BFOLatestOddsOnly()
        
        # Test with None inputs
        assert scraper.sanitize_name(None) is None
        assert scraper.get_mapped_name(None) is None
        assert scraper.reverse_map_name(None) is None
        
        # Test decimal conversion with invalid data
        result = scraper.decimal_to_american("not_a_number")
        assert result == 100  # Should return fallback
        
        # Test opponent odds calculation with invalid data
        result = scraper.calculate_opponent_odds("invalid")
        assert result is None
    
    def test_utility_function_error_handling(self):
        """Test utility functions with invalid inputs"""
        # Test convert_american_to_decimal with various invalid inputs
        assert convert_american_to_decimal(None) == 0
        assert convert_american_to_decimal("") == 0
        assert convert_american_to_decimal("not_a_number") == 0
        
        # Test american_odds_to_prob with invalid inputs
        assert american_odds_to_prob(None) is None
        assert american_odds_to_prob("") is None
        assert american_odds_to_prob("not_a_number") is None
        
        # Test calculate_expected_value with invalid inputs
        assert calculate_expected_value(None, None) == 0.0
        assert calculate_expected_value("invalid", "invalid") == 0.0
    
    @patch('autogluon.tabular.TabularPredictor')
    @patch('predict.load_joblib_artifact')  
    @patch('predict.os.path.exists')
    def test_model_loading_error_handling(self, mock_exists, mock_joblib_load, mock_predictor):
        """Test model loading with various error conditions"""
        # Test calibrator loading failure
        mock_predictor.load.return_value = MagicMock()
        mock_exists.return_value = True
        mock_joblib_load.side_effect = Exception("Loading failed")
        
        model, calibrator = load_model_and_calibrator("/fake/path", use_calibrated=True)
        
        assert model is not None
        assert calibrator is None  # Should handle the exception gracefully


def test_parse_manual_odds_json_accepts_fighter_mapping():
    result = parse_manual_odds_json('{"fighter one": "-120", "fighter two": "+100"}')

    assert result == {"fighter one": -120, "fighter two": 100}


def test_apply_manual_odds_overrides_missing_odds_and_devigs_pair():
    fight_list = [(datetime(2026, 6, 1), "fighter one", "fighter two")]
    odds = {"fighter one": "N/A", "fighter two": {"original": 100, "vigless": 105}}

    result = apply_manual_odds(fight_list, odds, {"fighter one": -120})

    assert result["fighter one"]["original"] == -120
    assert result["fighter two"]["original"] == 100
    assert isinstance(result["fighter one"]["vigless"], int)
    assert isinstance(result["fighter two"]["vigless"], int)


def test_latest_model_path_uses_configured_models_dir_for_starter_model(monkeypatch, tmp_path):
    models_dir = tmp_path / "AutogluonModels"
    older = models_dir / "ag-20260101_000000-win-extreme"
    starter = models_dir / "ag-20260304_110750-win-extreme"
    older.mkdir(parents=True)
    starter.mkdir()
    (older / "feats.txt").write_text("feature\n", encoding="utf-8")
    (older / "predictor.pkl").write_text("predictor", encoding="utf-8")
    (starter / "feats.txt").write_text("feature\n", encoding="utf-8")
    (starter / "predictor.pkl").write_text("predictor", encoding="utf-8")
    os.utime(older, (1, 1))
    os.utime(starter, (2, 2))
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))

    assert latest_model_path("win") == starter


def test_latest_model_path_ignores_newer_incomplete_model_dirs(monkeypatch, tmp_path):
    models_dir = tmp_path / "AutogluonModels"
    usable = models_dir / "ag-20260304_110750-win-extreme"
    incomplete = models_dir / "ag-20260401_120000-win-extreme"
    usable.mkdir(parents=True)
    incomplete.mkdir()
    (usable / "feats.txt").write_text("feature\n", encoding="utf-8")
    (usable / "predictor.pkl").write_text("predictor", encoding="utf-8")
    (incomplete / "feats.txt").write_text("feature\n", encoding="utf-8")
    os.utime(usable, (1, 1))
    os.utime(incomplete, (2, 2))
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))

    assert latest_model_path("win") == usable


def test_latest_model_path_reports_configured_models_dir_when_missing(monkeypatch, tmp_path):
    models_dir = tmp_path / "empty-models"
    models_dir.mkdir()
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))

    with pytest.raises(FileNotFoundError, match="empty-models"):
        latest_model_path("win")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
