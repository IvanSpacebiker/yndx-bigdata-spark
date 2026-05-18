package ru.yndx.bigdata.hypotheses;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.yndx.bigdata.pipeline.Datasets;
import ru.yndx.bigdata.stats.StatUtils;

import java.util.List;

import static org.apache.spark.sql.functions.*;

/**
 * H1 — Listen+ rate is significantly lower for recommended tracks (is_organic=0)
 *      than for organic tracks (is_organic=1), controlling for track length.
 *
 * Method: Mann-Whitney U (one-sided, alternative="less") on played_ratio_pct.
 *         Rank-biserial r as effect size.
 *         Listen+ rate per length_bucket for confound control.
 */
public class H1ListenPlus {

    private static final Logger log = LoggerFactory.getLogger(H1ListenPlus.class);
    private static final int SAMPLE_SIZE = 100_000;
    private static final long SEED = 42L;

    private final SparkSession spark;
    private final Datasets ds;

    public H1ListenPlus(SparkSession spark, Datasets ds) {
        this.spark = spark;
        this.ds    = ds;
    }

    public HypothesisResult run() {
        log.info("=== H1: Listen+ rate ===");

        Dataset<Row> listens = ds.listensEnriched
                .filter(col("played_ratio_pct").isNotNull());

        // ── Overall Listen+ rate per group ───────────────────────────────────
        log.info("Listen+ rate by is_organic:");
        listens.groupBy("is_organic")
               .agg(
                   count("*").alias("total"),
                   sum("listen_plus").alias("lp_count"),
                   avg("listen_plus").alias("lp_rate"),
                   avg("played_ratio_pct").alias("mean_ratio"),
                   functions.expr("percentile(played_ratio_pct, 0.5)").alias("median_ratio")
               )
               .orderBy("is_organic")
               .show();

        // ── Confound control: Listen+ by length_bucket × is_organic ──────────
        log.info("Listen+ rate by length_bucket × is_organic:");
        listens.groupBy("length_bucket", "is_organic")
               .agg(avg("listen_plus").alias("lp_rate"))
               .orderBy("length_bucket", "is_organic")
               .show();

        // ── Mann-Whitney U on sample ──────────────────────────────────────────
        // Collect samples for the two groups
        double[] recSample = sampleRatios(listens, 0);
        double[] orgSample = sampleRatios(listens, 1);

        // One-sided: alternative = rec < org
        double[] mw = StatUtils.mannWhitneyLess(recSample, orgSample);
        double uStat = mw[0], pValue = mw[1], rRB = mw[2];

        log.info("Mann-Whitney U: U={}, p={}, rank-biserial r={}", uStat, pValue, rRB);

        boolean confirmed = pValue < 0.05 && rRB < 0;   // rec is ranked lower
        String details = String.format(
                "Mann-Whitney U=%.2e, p=%.4e, rank-biserial r=%.4f (sample=%d per group)",
                uStat, pValue, rRB, SAMPLE_SIZE);

        log.info("H1 {}", confirmed ? "CONFIRMED" : "NOT CONFIRMED");
        return new HypothesisResult("H1",
                "Listen+ ниже для рекомендательных треков",
                confirmed, details);
    }

    private double[] sampleRatios(Dataset<Row> listens, int isOrganic) {
        List<Row> rows = listens
                .filter(col("is_organic").equalTo(isOrganic))
                .select("played_ratio_pct")
                .sample(false, computeFraction(listens, isOrganic, SAMPLE_SIZE), SEED)
                .limit(SAMPLE_SIZE)
                .collectAsList();

        double[] arr = new double[rows.size()];
        for (int i = 0; i < rows.size(); i++) {
			arr[i] = ((Number) rows.get(i).get(0)).doubleValue();
        }
        return arr;
    }

    /** Rough fraction to aim for SAMPLE_SIZE rows without a full count scan. */
    private double computeFraction(Dataset<Row> df, int isOrganic, int target) {
        long n = df.filter(col("is_organic").equalTo(isOrganic)).count();
        return n <= target ? 1.0 : (double) target / n * 1.2;  // 20% over-sample, then limit
    }
}
