package ru.yndx.bigdata.hypotheses;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.yndx.bigdata.pipeline.Datasets;
import ru.yndx.bigdata.pipeline.Preprocessing;
import ru.yndx.bigdata.stats.StatUtils;

import java.util.List;

import static org.apache.spark.sql.functions.*;

/**
 * H3 — Dislikes of recommended tracks are cancelled (undisliked) more often
 *       than dislikes of organic tracks.
 *
 * Algorithm:
 *   1. Enrich dislikes with abs_time_sec
 *   2. Enrich undislikes with abs_time_sec, rename to undislike_time
 *   3. Left-join on (uid, item_id); keep only undislikes AFTER the dislike
 *   4. Aggregate: was there at least one valid undislike per dislike?
 *   5. Chi-squared + OR + Bootstrap CI
 *   6. Median time-to-cancel (Mann-Whitney two-sided)
 */
public class H3CancellationRate {

    private static final Logger log = LoggerFactory.getLogger(H3CancellationRate.class);
    private static final int N_BOOTSTRAP = 2000;
    private static final long SEED = 42L;

    private final SparkSession spark;
    private final Datasets ds;

    public H3CancellationRate(SparkSession spark, Datasets ds) {
        this.spark = spark;
        this.ds    = ds;
    }

    public HypothesisResult run() {
        log.info("=== H3: Dislike cancellation rate ===");

        // 1+2. Enrich both sides
        Dataset<Row> dis = Preprocessing.reconstructTime(ds.dislikes)
                .select("uid", "item_id", "abs_time_sec", "is_organic");

        Dataset<Row> und = Preprocessing.reconstructTime(ds.undislikes).select(
				col("uid"),
				col("item_id"),
				col("abs_time_sec").alias("undislike_time")
		);

        // 3. Join and filter: undislike must be AFTER dislike
        Dataset<Row> merged = dis
                .join(und, new String[] {"uid", "item_id"}, "left")
                .withColumn("valid_cancel",
                        und.col("undislike_time").isNotNull()
                        .and(col("undislike_time").gt(col("abs_time_sec"))));

        // 4. Aggregate per (uid, item_id, dislike_time): was there a cancel?
        Dataset<Row> h3 = merged
                .groupBy("uid", "item_id", "abs_time_sec", "is_organic")
                .agg(
                    max(col("valid_cancel").cast("int")).alias("cancelled"),
                    min("undislike_time").alias("min_cancel_time")
                )
                .withColumn("cancelled", coalesce(col("cancelled"), lit(0)))
                .withColumn("time_to_cancel",
                        col("min_cancel_time").minus(col("abs_time_sec")));

        log.info("Cancellation rate by group:");
        h3.groupBy("is_organic")
          .agg(
              count("*").alias("total_dislikes"),
              sum("cancelled").alias("cancelled_count"),
              avg("cancelled").alias("cancellation_rate")
          )
          .orderBy("is_organic")
          .show();

        // 5. Chi-squared + OR
        long[][] ct = buildCT(h3);
        double[] chi = StatUtils.chiSquared2x2(ct);
        double chi2 = chi[0], pChi2 = chi[1], or = chi[2];

        // Bootstrap CI
        double[] recArr = collectBinary(h3, 0, "cancelled");
        double[] orgArr = collectBinary(h3, 1, "cancelled");
        double[] boot   = StatUtils.bootstrapDiffCI(recArr, orgArr, N_BOOTSTRAP, SEED);

        log.info("Chi-squared: χ²={}, p={}, OR={}", chi2, pChi2, or);
        log.info("Bootstrap CI: diff={}, 95% CI=[{}, {}]", boot[0], boot[1], boot[2]);

        // 6. Time-to-cancel comparison (only cancelled rows)
        Dataset<Row> cancelled = h3.filter(col("cancelled").equalTo(1));
        double[] recTime = collectDouble(cancelled, 0, "time_to_cancel");
        double[] orgTime = collectDouble(cancelled, 1, "time_to_cancel");
        if (recTime.length > 1 && orgTime.length > 1) {
            double[] mw = StatUtils.mannWhitneyTwoSided(recTime, orgTime);
            log.info("Time-to-cancel Mann-Whitney: p={}", mw[1]);
            log.info("  Median rec={}, org={}",
                    median(recTime), median(orgTime));
        }

        boolean confirmed = pChi2 < 0.05 && or > 1.0;
        String details = String.format(
                "χ²=%.2f, p=%.4e, OR=%.4f | Bootstrap CI=[%+.5f, %+.5f]",
                chi2, pChi2, or, boot[1], boot[2]);

        return new HypothesisResult("H3",
                "Дизлайки рекомендательных чаще отменяются",
                confirmed, details);
    }

    private long[][] buildCT(Dataset<Row> h3) {
        List<Row> rows = h3.groupBy("is_organic", "cancelled")
                           .count()
                           .orderBy("is_organic", "cancelled")
                           .collectAsList();
        long[][] ct = new long[2][2];
        for (Row r : rows) {
            int org  = ((Number) r.get(0)).intValue();
            int canc = ((Number) r.get(1)).intValue();
            ct[org][canc] = ((Number) r.get(2)).longValue();
        }
        return ct;
    }

    private double[] collectBinary(Dataset<Row> df, int isOrganic, String colName) {
        List<Row> rows = df.filter(col("is_organic").equalTo(isOrganic))
                           .select(colName).collectAsList();
        double[] arr = new double[rows.size()];
        for (int i = 0; i < rows.size(); i++) arr[i] = ((Number) rows.get(i).get(0)).intValue();
        return arr;
    }

    private double[] collectDouble(Dataset<Row> df, int isOrganic, String colName) {
        List<Row> rows = df.filter(col("is_organic").equalTo(isOrganic))
                           .select(colName)
                           .filter(col(colName).isNotNull())
                           .collectAsList();
        double[] arr = new double[rows.size()];
        for (int i = 0; i < rows.size(); i++) arr[i] = ((Number) rows.get(i).get(0)).doubleValue();
        return arr;
    }

    private double median(double[] arr) {
        double[] sorted = arr.clone();
        java.util.Arrays.sort(sorted);
        int n = sorted.length;
        return n % 2 == 0 ? (sorted[n/2-1] + sorted[n/2]) / 2.0 : sorted[n/2];
    }
}
