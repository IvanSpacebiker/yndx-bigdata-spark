package ru.yndx.bigdata.stats;

import org.apache.commons.math3.stat.inference.ChiSquareTest;
import org.apache.commons.math3.stat.inference.MannWhitneyUTest;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.util.FastMath;

import java.util.Random;

/**
 * Thin wrappers around Apache Commons Math 3 + custom bootstrap.
 * All methods are stateless and thread-safe.
 */
public final class StatUtils {

    private StatUtils() {}

    // ── Mann-Whitney U ────────────────────────────────────────────────────────

    /**
     * One-sided test: H1 = x < y (alternative = "less").
     * Returns [U-statistic, p-value, rank-biserial r].
     */
    public static double[] mannWhitneyLess(double[] x, double[] y) {
        MannWhitneyUTest test = new MannWhitneyUTest();
        double u = test.mannWhitneyU(x, y);
        double p = test.mannWhitneyUTest(x, y);   // two-sided from commons-math
        // Convert to one-sided (less): if U is in the lower tail, p_one = p/2, else 1-p/2
        double n1 = x.length, n2 = y.length;
        double uMax = n1 * n2;
        double pOneSided = u <= uMax / 2 ? p / 2.0 : 1.0 - p / 2.0;
        double rRankBiserial = 1.0 - (2.0 * u) / (n1 * n2);
        return new double[]{u, pOneSided, rRankBiserial};
    }

    /**
     * Two-sided test. Returns [U, p].
     */
    public static double[] mannWhitneyTwoSided(double[] x, double[] y) {
        MannWhitneyUTest test = new MannWhitneyUTest();
        double u = test.mannWhitneyU(x, y);
        double p = test.mannWhitneyUTest(x, y);
        return new double[]{u, p};
    }

    // ── Chi-squared ──────────────────────────────────────────────────────────

    /**
     * Chi-squared test of independence on a 2×2 contingency table.
     * table[i][j]: row i, col j (i=group, j=outcome).
     * Returns [chi2, p-value, odds-ratio].
     */
    public static double[] chiSquared2x2(long[][] table) {
        long[][] copy = new long[][]{table[0].clone(), table[1].clone()};
        ChiSquareTest test = new ChiSquareTest();
        double chi2 = test.chiSquare(copy);
        double p    = test.chiSquareTest(copy);
        // OR = (a*d) / (b*c)
        double a = table[0][1], b = table[0][0];
        double c = table[1][1], d = table[1][0];
        double or = (b * c > 0) ? (a * d) / (b * c) : Double.NaN;
        return new double[]{chi2, p, or};
    }

    // ── Bootstrap CI ─────────────────────────────────────────────────────────

    /**
     * Bootstrap 95% CI for the difference in means: mean(a) - mean(b).
     * Returns [observed_diff, ci_low, ci_high].
     */
    public static double[] bootstrapDiffCI(double[] a, double[] b,
                                           int nBootstrap, long seed) {
        Random rng = new Random(seed);
        double[] diffs = new double[nBootstrap];
        for (int i = 0; i < nBootstrap; i++) {
            diffs[i] = sampleMean(a, rng) - sampleMean(b, rng);
        }
        java.util.Arrays.sort(diffs);
        double ciLow  = diffs[(int)(nBootstrap * 0.025)];
        double ciHigh = diffs[(int)(nBootstrap * 0.975)];
        double obs    = mean(a) - mean(b);
        return new double[]{obs, ciLow, ciHigh};
    }

    private static double sampleMean(double[] arr, Random rng) {
        double sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum += arr[rng.nextInt(arr.length)];
        }
        return sum / arr.length;
    }

    public static double mean(double[] arr) {
        double s = 0;
        for (double v : arr) s += v;
        return arr.length == 0 ? 0 : s / arr.length;
    }

    // ── Fisher z-test ─────────────────────────────────────────────────────────

    /**
     * Compares two Spearman/Pearson correlations via Fisher z-transformation.
     * Returns [z-statistic, two-sided p-value].
     */
    public static double[] fisherZTest(double r1, double r2, int n1, int n2) {
        double z1 = FastMath.atanh(clip(r1, -0.9999, 0.9999));
        double z2 = FastMath.atanh(clip(r2, -0.9999, 0.9999));
        double se = Math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3));
        double z  = (z1 - z2) / se;
        NormalDistribution normal = new NormalDistribution();
        double p = 2.0 * (1.0 - normal.cumulativeProbability(Math.abs(z)));
        return new double[]{z, p};
    }

	private static double clip(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    // ── Spearman rho ─────────────────────────────────────────────────────────

    /**
     * Spearman rank correlation for two arrays of the same length.
     */
    public static double[] spearman(double[] x, double[] y) {
        org.apache.commons.math3.stat.correlation.SpearmansCorrelation sc =
                new org.apache.commons.math3.stat.correlation.SpearmansCorrelation();
        double rho = sc.correlation(x, y);
        // p-value approximation via t-distribution
        int n = x.length;
        double t = rho * Math.sqrt((n - 2.0) / (1.0 - rho * rho));
        org.apache.commons.math3.distribution.TDistribution td =
                new org.apache.commons.math3.distribution.TDistribution(n - 2);
        double p = 2.0 * (1.0 - td.cumulativeProbability(Math.abs(t)));
        return new double[]{rho, p};
    }
}
