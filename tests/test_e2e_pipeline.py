"""End-to-end pipeline tests for untargeted-metabolomics.

Organised in three tiers:

  1. Unit — pure-function tests for library search and blank subtraction
     helpers; no I/O, no heavy dependencies, always fast.

  2. Feature-finding — imports ``run_feature_finding`` from step 02 and runs
     it against one real mzML file from data/raw/.  Marked ``slow`` and
     ``requires_mzml``; skipped if the file is absent or pyopenms is missing.

  3. Integration — runs the full step 01-10 pipeline end-to-end as subprocess
     calls from the project root.  Marked ``integration``; needs all deps and
     the data/raw/ mzML + GNPS subset files.

Run tiers selectively:
  pytest                            # unit only
  pytest -m slow                    # feature-finding
  pytest -m integration             # full pipeline
  pytest -m "slow or integration"   # all real-data tests
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"

# Real mzML files present in the repo
MIX_01   = DATA_RAW / "MIX_01.mzML"
BLANK_01 = DATA_RAW / "BLANK_01.mzML"

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests that load real mzML files")
    config.addinivalue_line("markers", "integration: full subprocess pipeline tests")


# ===========================================================================
# Tier 1: Unit tests — pure functions, no I/O
# ===========================================================================

class TestCosineLibrarySearch:
    """Tests for cosine_similarity and normalize from script 09."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from importlib import import_module
        mod = import_module("09_library_search")
        self.cosine = mod.cosine_similarity
        self.normalize = mod.normalize
        self.parse_pepmass = mod.parse_pepmass

    # normalize
    def test_normalize_unit_vector_is_unchanged(self):
        v = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(self.normalize(v), v)

    def test_normalize_scales_to_unit_l2(self):
        v = np.array([3.0, 4.0])
        out = self.normalize(v)
        assert abs(np.linalg.norm(out) - 1.0) < 1e-9

    def test_normalize_zero_vector_returns_zero(self):
        v = np.zeros(4)
        np.testing.assert_array_equal(self.normalize(v), v)

    # cosine_similarity
    def test_identical_spectra_score_one(self):
        mz  = np.array([100.0, 150.0, 200.0])
        ints = np.array([1000.0, 500.0, 250.0])
        score = self.cosine(mz, ints, mz, ints, mz_tolerance=0.01)
        assert abs(score - 1.0) < 1e-6

    def test_orthogonal_spectra_score_zero(self):
        mz_a  = np.array([100.0, 200.0])
        int_a = np.array([1.0, 1.0])
        mz_b  = np.array([300.0, 400.0])
        int_b = np.array([1.0, 1.0])
        score = self.cosine(mz_a, int_a, mz_b, int_b, mz_tolerance=0.01)
        assert score == 0.0

    def test_partial_overlap_score_between_zero_and_one(self):
        mz_a  = np.array([100.0, 150.0, 200.0])
        int_a = np.array([1000.0, 500.0, 250.0])
        mz_b  = np.array([100.0, 175.0, 250.0])  # one matching peak at 100
        int_b = np.array([800.0, 400.0, 200.0])
        score = self.cosine(mz_a, int_a, mz_b, int_b, mz_tolerance=0.01)
        assert 0.0 < score < 1.0

    def test_score_within_tolerance_matches(self):
        mz_a  = np.array([100.0])
        int_a = np.array([1.0])
        mz_b  = np.array([100.005])   # 5 mDa off — within 10 mDa tolerance
        int_b = np.array([1.0])
        score = self.cosine(mz_a, int_a, mz_b, int_b, mz_tolerance=0.01)
        assert score > 0.9

    def test_score_outside_tolerance_does_not_match(self):
        mz_a  = np.array([100.0])
        int_a = np.array([1.0])
        mz_b  = np.array([100.05])   # 50 mDa off — outside 10 mDa tolerance
        int_b = np.array([1.0])
        score = self.cosine(mz_a, int_a, mz_b, int_b, mz_tolerance=0.01)
        assert score == 0.0

    def test_empty_spectrum_a_returns_zero(self):
        score = self.cosine(np.array([]), np.array([]), np.array([100.0]), np.array([1.0]), 0.01)
        assert score == 0.0

    def test_empty_spectrum_b_returns_zero(self):
        score = self.cosine(np.array([100.0]), np.array([1.0]), np.array([]), np.array([]), 0.01)
        assert score == 0.0

    def test_score_symmetric(self):
        mz_a  = np.array([100.0, 150.0])
        int_a = np.array([500.0, 200.0])
        mz_b  = np.array([100.0, 200.0])
        int_b = np.array([300.0, 100.0])
        s1 = self.cosine(mz_a, int_a, mz_b, int_b, 0.01)
        s2 = self.cosine(mz_b, int_b, mz_a, int_a, 0.01)
        assert abs(s1 - s2) < 1e-9

    # parse_pepmass
    def test_parse_pepmass_float(self):
        assert self.parse_pepmass(123.456) == pytest.approx(123.456)

    def test_parse_pepmass_string_with_intensity(self):
        assert self.parse_pepmass("200.5 1000") == pytest.approx(200.5)

    def test_parse_pepmass_none(self):
        assert self.parse_pepmass(None) is None


class TestBlankSubtraction:
    """Tests for match_known and adduct matching from script 04."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from importlib import import_module
        mod = import_module("04_blank_subtract")
        self.match_known = mod.match_known
        self.KNOWN_ADDUCTS = mod.KNOWN_ADDUCTS

    def test_exact_mass_match_returns_true(self):
        masses = [180.0634]   # glucose monoisotopic mass
        mz = 180.0634          # neutral [M]
        hit, mass, adduct, ppm = self.match_known(mz, masses, ppm=10.0)
        assert hit is True
        assert mass == pytest.approx(180.0634)
        assert adduct == "M"

    def test_protonated_adduct_match(self):
        masses = [180.0634]
        mz = 180.0634 + 1.007276   # [M+H]+
        hit, mass, adduct, ppm = self.match_known(mz, masses, ppm=10.0)
        assert hit is True
        assert adduct == "M+H"

    def test_sodium_adduct_match(self):
        masses = [180.0634]
        mz = 180.0634 + 22.989218   # [M+Na]+
        hit, mass, adduct, ppm = self.match_known(mz, masses, ppm=10.0)
        assert hit is True
        assert adduct == "M+Na"

    def test_no_match_returns_false(self):
        masses = [180.0634]
        mz = 500.0                   # far from glucose
        hit, _, _, _ = self.match_known(mz, masses, ppm=10.0)
        assert hit is False

    def test_empty_masses_returns_false(self):
        hit, _, _, _ = self.match_known(180.0, [], ppm=10.0)
        assert hit is False

    def test_ppm_threshold_respected(self):
        masses = [180.0634]
        # 20 ppm off from neutral — should fail at 10 ppm
        mz = 180.0634 * (1 + 20e-6)
        hit_tight, _, _, _ = self.match_known(mz, masses, ppm=10.0)
        hit_loose, _, _, _ = self.match_known(mz, masses, ppm=25.0)
        assert not hit_tight
        assert hit_loose

    def test_best_match_returned_for_multiple_masses(self):
        masses = [180.0634, 350.0]
        # mz matches 180.0634 exactly, not 350.0
        hit, mass, adduct, ppm = self.match_known(180.0634, masses, ppm=10.0)
        assert hit is True
        assert mass == pytest.approx(180.0634)
        assert ppm < 1.0

    def test_all_adducts_defined(self):
        """Ensure the KNOWN_ADDUCTS dict has the expected entries."""
        assert "M+H" in self.KNOWN_ADDUCTS
        assert "M+Na" in self.KNOWN_ADDUCTS
        assert "M+K" in self.KNOWN_ADDUCTS
        assert "M+NH4" in self.KNOWN_ADDUCTS


# ===========================================================================
# Tier 2: Feature finding on real mzML (slow — skipped if file absent)
# ===========================================================================

@pytest.mark.slow
class TestFeatureFinding:
    """Run pyopenms feature finding on a real MIX mzML file."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        if not MIX_01.exists():
            pytest.skip(f"mzML not found: {MIX_01}")
        try:
            import pyopenms  # noqa: F401
        except ImportError:
            pytest.skip("pyopenms not installed")
        from importlib import import_module
        mod = import_module("02_feature_finding")
        self.run_feature_finding = mod.run_feature_finding

    def _minimal_cfg(self) -> dict:
        return {
            "noise_threshold_int": 5000.0,
            "mass_error_ppm": 10.0,
            "chrom_fwhm": 5.0,
            "chrom_peak_snr": 3.0,
            "min_trace_length": 3.0,
            "max_trace_length": -1.0,
        }

    def test_feature_finding_returns_feature_map(self):
        import pyopenms as oms
        fmap, traces = self.run_feature_finding(MIX_01, self._minimal_cfg())
        assert isinstance(fmap, oms.FeatureMap)

    def test_feature_map_has_features(self):
        fmap, _ = self.run_feature_finding(MIX_01, self._minimal_cfg())
        assert len(fmap) > 0, "Expected at least one feature in a real MIX file"

    def test_features_have_valid_mz_and_rt(self):
        fmap, _ = self.run_feature_finding(MIX_01, self._minimal_cfg())
        for feat in fmap:
            assert feat.getMZ() > 0.0
            assert feat.getRT() > 0.0

    def test_blank_produces_fewer_features_than_mix(self):
        if not BLANK_01.exists():
            pytest.skip(f"Blank mzML not found: {BLANK_01}")
        mix_map,   _ = self.run_feature_finding(MIX_01,   self._minimal_cfg())
        blank_map, _ = self.run_feature_finding(BLANK_01, self._minimal_cfg())
        # Biological samples should have more features than blank injections
        assert len(mix_map) >= len(blank_map), (
            f"MIX ({len(mix_map)} features) should have >= blank ({len(blank_map)} features)"
        )


# ===========================================================================
# Tier 3: Full subprocess pipeline (integration — needs all data + deps)
# ===========================================================================

def _run_step(script_name: str, cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a pipeline step script via the project's Python interpreter."""
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.integration
class TestFullPipeline:
    """Run all pipeline steps end-to-end using real data in data/raw/."""

    @pytest.fixture(autouse=True, scope="class")
    def _check_data(self):
        if not MIX_01.exists() or not BLANK_01.exists():
            pytest.skip("mzML files not found in data/raw/ — skipping integration tests")
        try:
            import pyopenms  # noqa: F401
        except ImportError:
            pytest.skip("pyopenms not installed")

    def test_step_01_validate_inputs(self):
        result = _run_step("01_validate_inputs.py", PROJECT_ROOT)
        assert result.returncode == 0, f"Step 01 failed:\n{result.stderr}"
        assert (DATA_INTERIM / "sample_metadata.tsv").exists()

    def test_step_02_feature_finding(self):
        result = _run_step("02_feature_finding.py", PROJECT_ROOT, timeout=600)
        assert result.returncode == 0, f"Step 02 failed:\n{result.stderr}"
        interim_featuremaps = list(DATA_INTERIM.glob("features_*.featureXML"))
        assert len(interim_featuremaps) > 0, "No feature maps written by step 02"

    def test_step_03_align_features(self):
        result = _run_step("03_align_features.py", PROJECT_ROOT)
        assert result.returncode == 0, f"Step 03 failed:\n{result.stderr}"
        assert (DATA_INTERIM / "feature_groups.tsv").exists()

    def test_step_04_blank_subtract(self):
        result = _run_step("04_blank_subtract.py", PROJECT_ROOT)
        assert result.returncode == 0, f"Step 04 failed:\n{result.stderr}"
        filtered = (DATA_PROC / "feature_groups_filtered.tsv")
        assert filtered.exists()

    def test_step_05_adduct_filter(self):
        result = _run_step("05_adduct_filter.py", PROJECT_ROOT)
        assert result.returncode == 0, f"Step 05 failed:\n{result.stderr}"

    def test_step_06_extract_ms2(self):
        result = _run_step("06_extract_ms2.py", PROJECT_ROOT, timeout=600)
        assert result.returncode == 0, f"Step 06 failed:\n{result.stderr}"
        assert (DATA_INTERIM / "ms2_spectra.parquet").exists()

    def test_step_07_link_ms2_features(self):
        result = _run_step("07_link_ms2_features.py", PROJECT_ROOT)
        assert result.returncode == 0, f"Step 07 failed:\n{result.stderr}"

    def test_step_08_build_library_index(self):
        gnps_mgf = DATA_RAW / "GNPS_SUBSET.mgf.zip"
        if not gnps_mgf.exists():
            pytest.skip("GNPS_SUBSET.mgf.zip not found — skipping library index build")
        result = _run_step("08_build_library_index.py", PROJECT_ROOT, timeout=600)
        assert result.returncode == 0, f"Step 08 failed:\n{result.stderr}"
        assert (DATA_INTERIM / "gnps_library.sqlite").exists()

    def test_step_09_library_search(self):
        if not (DATA_INTERIM / "gnps_library.sqlite").exists():
            pytest.skip("GNPS sqlite not built — run step 08 first")
        result = _run_step("09_library_search.py", PROJECT_ROOT)
        assert result.returncode == 0, f"Step 09 failed:\n{result.stderr}"
        assert (DATA_PROC / "library_hits.tsv").exists()

    def test_step_10_report_tables(self):
        result = _run_step("10_report_tables.py", PROJECT_ROOT)
        assert result.returncode == 0, f"Step 10 failed:\n{result.stderr}"

    def test_library_hits_have_expected_columns(self):
        hits_path = DATA_PROC / "library_hits.tsv"
        if not hits_path.exists():
            pytest.skip("library_hits.tsv not yet generated")
        import pandas as pd
        hits = pd.read_csv(hits_path, sep="\t")
        required_cols = {"precursor_mz", "cosine_score"}
        assert required_cols <= set(hits.columns), (
            f"Missing columns: {required_cols - set(hits.columns)}"
        )

    def test_library_hits_cosine_scores_in_valid_range(self):
        hits_path = DATA_PROC / "library_hits.tsv"
        if not hits_path.exists():
            pytest.skip("library_hits.tsv not yet generated")
        import pandas as pd
        hits = pd.read_csv(hits_path, sep="\t")
        assert (hits["cosine_score"] >= 0.0).all()
        assert (hits["cosine_score"] <= 1.0).all()
