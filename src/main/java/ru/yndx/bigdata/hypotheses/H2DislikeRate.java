package ru.yndx.bigdata.hypotheses;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.yndx.bigdata.pipeline.Datasets;
import ru.yndx.bigdata.stats.StatUtils;

import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.*;

/**
 * H2 — P(dislike | listen) is significantly higher for recommended tracks.
 *
 * Algorithm (mirrors notebook):
 *   1. Aggregate listens by (uid, item_id) → is_organic (last), n_listens
 *   2. Left-join dislikes.distinct(uid, item_id) → disliked flag
 *   3. Chi-squared + Odds Ratio
 *   4. Bootstrap 95% CI for difference in dislike rates
 */
public class H2DislikeRate {

    private static final Logger log = LoggerFactory.getLogger(H2DislikeRate.class);
    private static final int N_BOOTSTRAP = 2000;
    private static final long SEED = 42L;

    private final SparkSession spark;
    private final Datasets ds;

    public H2DislikeRate(SparkSession spark, Datasets ds) {
        this.spark = spark;
        this.ds    = ds;
    }

	public HypothesisResult run() {
		log.info("=== H2: P(dislike | listen) ===");

		// 1. Aggregate listens: one row per (uid, item_id)
		Dataset<Row> listenAgg = ds.listensEnriched
				.groupBy("uid", "item_id")
				.agg(
						last("is_organic").alias("is_organic"),
						count("*").alias("n_listens"),
						max("listen_plus").alias("listen_plus")
				);

		// 2. Dislike flag
		Dataset<Row> dislikeFlag = ds.dislikes
				.select("uid", "item_id")
				.distinct()
				.withColumn("disliked", lit(1));

		Dataset<Row> h2 = listenAgg
				.join(dislikeFlag, new String[]{"uid", "item_id"}, "left")
				.withColumn("disliked", coalesce(col("disliked"), lit(0)));

		// 3. All aggregation on Spark side — collect only 4 summary rows
		log.info("P(dislike) by group:");
		Dataset<Row> summary = h2.groupBy("is_organic")
				.agg(
						count("*").alias("pairs"),
						sum("disliked").alias("disliked_count"),
						avg("disliked").alias("dislike_rate")
				)
				.orderBy("is_organic");
		summary.show();

		// Build 2×2 contingency table from summary (2 rows, not millions)
		List<Row> sumRows = summary.collectAsList();
		long[][] ct = new long[2][2];
		double rateRec = 0, rateOrg = 0;
		long nRec = 0, nOrg = 0;

		for (Row r : sumRows) {
			int org        = ((Number) r.get(0)).intValue();
			long total     = r.getLong(1);
			long disliked  = r.getLong(2);
			long notDislik = total - disliked;
			ct[org][0] = notDislik;
			ct[org][1] = disliked;
			if (org == 0) { rateRec = r.getDouble(3); nRec = total; }
			else          { rateOrg = r.getDouble(3); nOrg = total; }
		}

		// 4. Chi-squared + OR
		double[] chiResult = StatUtils.chiSquared2x2(ct);
		double chi2 = chiResult[0], pChi2 = chiResult[1], or = chiResult[2];
		log.info("Chi-squared: χ²={}, p={}, OR={}", chi2, pChi2, or);

		// 5. Bootstrap CI from contingency table counts — no collect of raw rows needed
		//    We resample from Binomial(n, p) for each group using the observed rates
		double obsDiff = rateRec - rateOrg;
		double[] bootDiffs = new double[N_BOOTSTRAP];
		java.util.Random rng = new java.util.Random(SEED);
		for (int b = 0; b < N_BOOTSTRAP; b++) {
			// Resample counts from binomial
			long recDis = 0, orgDis = 0;
			for (long i = 0; i < nRec; i++) if (rng.nextDouble() < rateRec) recDis++;
			for (long i = 0; i < nOrg; i++) if (rng.nextDouble() < rateOrg) orgDis++;
			bootDiffs[b] = (double) recDis / nRec - (double) orgDis / nOrg;
		}
		Arrays.sort(bootDiffs);
		double ciLow  = bootDiffs[(int)(N_BOOTSTRAP * 0.025)];
		double ciHigh = bootDiffs[(int)(N_BOOTSTRAP * 0.975)];

		log.info("Bootstrap CI: diff={}, 95% CI=[{}, {}]", obsDiff, ciLow, ciHigh);

		boolean confirmed = pChi2 < 0.05 && or > 1.0 && ciLow > 0;
		String details = String.format(
				"χ²=%.2f, p=%.4e, OR=%.4f | Bootstrap CI=[%+.5f, %+.5f]",
				chi2, pChi2, or, ciLow, ciHigh);

		return new HypothesisResult("H2",
				"P(dislike) выше для рекомендательных треков",
				confirmed, details);
	}

}
