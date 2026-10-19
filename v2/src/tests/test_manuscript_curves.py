from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "v2/src/scripts"))
import analyze_supplement_v2 as figures


class ManuscriptCurveTest(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = figures.plt.subplots()

    def tearDown(self):
        figures.plt.close(self.fig)

    def test_curve_preserves_recorded_positions_and_sample_sd(self):
        x = np.array([1, 2, 3])
        values = np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.5, 0.6, 0.7]])
        result = figures.plot_band(self.ax, x, values, "Recorded data", "#0072B2")
        np.testing.assert_array_equal(self.ax.lines[0].get_xdata(), x)
        np.testing.assert_array_equal(self.ax.lines[0].get_ydata(), values.mean(0))
        np.testing.assert_array_equal(result["sd"], values.std(0, ddof=1))
        self.assertEqual(self.ax.lines[0].get_marker(), "None")

    def test_nonmonotone_positions_are_rejected(self):
        with self.assertRaises(AssertionError):
            figures.plot_band(self.ax, [1, 3, 2], np.ones((3, 3)), "Invalid order", "#0072B2")

    def test_nonpositive_logarithmic_data_are_rejected(self):
        with self.assertRaises(AssertionError):
            figures.plot_band(self.ax, [1, 2], np.zeros((3, 2)),
                              "Invalid log data", "#0072B2", positive=True)


if __name__ == "__main__":
    unittest.main()
