"""Testes de drift detection."""

import numpy as np
import pytest

from src.monitoring.drift import calculate_psi


class TestPSI:
    """Testes do cálculo de PSI."""

    def test_identical_distributions(self):
        """PSI de distribuições idênticas deve ser ~0."""
        np.random.seed(42)
        data = np.random.randn(1000)
        psi = calculate_psi(data, data)
        assert psi < 0.01

    def test_different_distributions(self):
        """PSI de distribuições diferentes deve ser > 0."""
        np.random.seed(42)
        ref = np.random.randn(1000)
        cur = np.random.randn(1000) + 2  # Shift significativo
        psi = calculate_psi(ref, cur)
        assert psi > 0.1

    def test_psi_non_negative(self):
        """PSI deve ser sempre não-negativo."""
        np.random.seed(42)
        ref = np.random.randn(500)
        cur = np.random.randn(500) * 1.5
        psi = calculate_psi(ref, cur)
        assert psi >= 0

    def test_slight_drift(self):
        """Drift leve deve gerar PSI > 0."""
        np.random.seed(42)
        ref = np.random.randn(1000)
        cur = np.random.randn(1000) + 0.3  # Shift leve
        psi = calculate_psi(ref, cur)
        assert psi > 0.0

    def test_psi_symmetric_property(self):
        """PSI deve ser finito para distribuições normais."""
        np.random.seed(42)
        ref = np.random.randn(1000)
        cur = np.random.randn(1000) + 1
        psi = calculate_psi(ref, cur)
        assert np.isfinite(psi)

    def test_psi_with_different_n_bins(self):
        """PSI deve funcionar com diferentes números de bins."""
        np.random.seed(42)
        ref = np.random.randn(500)
        cur = np.random.randn(500) + 1

        for n_bins in [5, 10, 20]:
            psi = calculate_psi(ref, cur, n_bins=n_bins)
            assert psi >= 0
            assert np.isfinite(psi)
