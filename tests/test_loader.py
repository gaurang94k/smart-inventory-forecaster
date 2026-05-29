"""Tests for src/data/loader.py."""

import pytest
import pandas as pd

import src.data.loader as loader


def test_load_sales_sample_rows():
    df = loader.load_sales(sample_skus=10)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 10


def test_load_calendar_date_dtype():
    cal = loader.load_calendar()
    assert isinstance(cal, pd.DataFrame)
    assert pd.api.types.is_datetime64_any_dtype(cal["date"])


class TestFileNotFoundErrors:
    """All three loaders must raise FileNotFoundError when DATA_DIR is fake."""

    @pytest.fixture(autouse=True)
    def patch_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(loader, "DATA_DIR", tmp_path / "nonexistent")

    def test_load_sales_missing(self):
        with pytest.raises(FileNotFoundError):
            loader.load_sales()

    def test_load_calendar_missing(self):
        with pytest.raises(FileNotFoundError):
            loader.load_calendar()

    def test_load_prices_missing(self):
        with pytest.raises(FileNotFoundError):
            loader.load_prices()
