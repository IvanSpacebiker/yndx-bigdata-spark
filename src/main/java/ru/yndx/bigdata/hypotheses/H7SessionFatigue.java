package ru.yndx.bigdata.hypotheses;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.yndx.bigdata.pipeline.Datasets;
import ru.yndx.bigdata.pipeline.Preprocessing;
import ru.yndx.bigdata.stats.StatUtils;

import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.*;

/**
 * H7 — In sessions where >50% of tracks are recommended,
 *       P(dislike) in the last third is higher than in the first third.
 *
 * Session boundary: > 30-minute gap between consecutive events of the same user.
 * Source: multi_event table (all event types in one table).
 */
public class H7SessionFatigue {

    private static final Logger log = LoggerFactory.getLogger(H7SessionFatigue.class);
    /** 30 min = 360 × 5-second units, but abs_time_sec is already in seconds */
    private static final long SESSION_GAP_SEC = 30 * 60;
    private static final int MIN_SESSION_LEN = 6;

    private final SparkSession spark;
    private final Datasets ds;

    public H7SessionFatigue(SparkSession spark, Datasets ds) {
        this.spark = spark;
        this.ds    = ds;
    }

	public HypothesisResult run() {
		log.info("=== H7: P(dislike) fatigue in recommendation-dominated sessions ===");

		// 1. Prune to only columns needed — removes unrelated window columns from plan
		Dataset<Row> me = Preprocessing.reconstructTime(ds.multiEvent)
				.select("uid", "abs_time_sec", "is_organic", "event_type")
				.sort("uid", "abs_time_sec");

		// 2. Session boundaries
		WindowSpec wByUser = Window.partitionBy("uid").orderBy("abs_time_sec");
		me = me
				.withColumn("prev_time", lag("abs_time_sec", 1).over(wByUser))
				.withColumn("time_delta",
						coalesce(col("abs_time_sec").minus(col("prev_time")), lit(0L)))
				.withColumn("new_session",
						when(col("time_delta").gt(SESSION_GAP_SEC), 1).otherwise(0))
				.withColumn("session_id",
						sum("new_session").over(wByUser));

		// Checkpoint to cut the lineage before more windows stack on top
		me = me.drop("prev_time", "time_delta", "new_session");
		me.cache();
		me.count(); // materialize

		// 3. Session length and position
		WindowSpec wBySession = Window.partitionBy("uid", "session_id").orderBy("abs_time_sec");
		me = me
				.withColumn("pos_in_session", row_number().over(wBySession).minus(1))
				.withColumn("session_len",
						count("*").over(Window.partitionBy("uid", "session_id")))
				.filter(col("session_len").geq(MIN_SESSION_LEN));

		// 4. Position third
		me = me.withColumn("pos_rel",
						col("pos_in_session").divide(col("session_len")).cast("double"))
				.withColumn("pos_third",
						when(col("pos_rel").lt(1.0 / 3), "beginning")
								.when(col("pos_rel").lt(2.0 / 3), "middle")
								.otherwise("end"))
				.withColumn("is_dislike",
						when(col("event_type").equalTo("dislike"), 1).otherwise(0));

		// 5. Rec share per session — Fix: count only listen events (where is_organic is not null)
		//    Non-listen events (dislike, skip) have null is_organic; exclude them from share calc
		Dataset<Row> recShare = me
				.filter(col("is_organic").isNotNull())   // ← key fix
				.groupBy("uid", "session_id")
				.agg(avg(when(col("is_organic").equalTo(0), 1).otherwise(0))
						.alias("rec_share"));

		me = me.join(recShare, new String[]{"uid", "session_id"})
				.withColumn("rec_dominated", col("rec_share").gt(0.5));

		me.cache();
		me.count(); // materialize before show + aggregations

		// 6. P(dislike) by third × session type
		log.info("P(dislike) by session third × type:");
		me.groupBy("rec_dominated", "pos_third")
				.agg(avg("is_dislike").alias("p_dislike"), count("*").alias("n"))
				.orderBy("rec_dominated", "pos_third")
				.show(20);

		// 7. Fix: compute stats fully on Spark side — never collect raw rows to driver
		Dataset<Row> stats = me
				.filter(col("rec_dominated").equalTo(true))
				.filter(col("pos_third").isin("beginning", "end"))
				.groupBy("pos_third")
				.agg(
						avg("is_dislike").alias("p_dislike"),
						count("*").alias("n"),
						stddev("is_dislike").alias("sd")
				);

		stats.show();
		List<Row> statRows = stats.collectAsList(); // only 2 rows — safe

		double pEnd = Double.NaN, pBegin = Double.NaN;
		long nEnd = 0, nBegin = 0;
		double sdEnd = 0, sdBegin = 0;

		for (Row r : statRows) {
			String third = r.getString(0);
			if ("end".equals(third))       { pEnd   = r.getDouble(1); nEnd   = r.getLong(2); sdEnd   = r.getDouble(3); }
			if ("beginning".equals(third)) { pBegin = r.getDouble(1); nBegin = r.getLong(2); sdBegin = r.getDouble(3); }
		}

		double pValue = Double.NaN;
		if (!Double.isNaN(pEnd) && !Double.isNaN(pBegin) && nEnd > 0 && nBegin > 0) {
			double pooled = (pEnd * nEnd + pBegin * nBegin) / (nEnd + nBegin);
			double se = Math.sqrt(pooled * (1 - pooled) * (1.0 / nEnd + 1.0 / nBegin));
			double z = (pEnd - pBegin) / se;
			// Two-sided p via normal CDF approximation (Abramowitz & Stegun)
			pValue = 2 * normalCdfComplement(Math.abs(z));
			log.info("Z-test (end vs beginning, rec sessions): z={}, p={}", z, pValue);
		}

		boolean confirmed = !Double.isNaN(pValue) && pValue < 0.05 && pEnd > pBegin;

		String details = String.format(
				"Two-prop z-test p=%.4e | P(dislike) end=%.5f (n=%d), beginning=%.5f (n=%d)",
				pValue, pEnd, nEnd, pBegin, nBegin);

		return new HypothesisResult("H7",
				"Усталость: P(dislike) растёт к концу рек-сессий",
				confirmed, details);
	}

	/** Normal survival function approximation (error < 1.5e-7) */
	private static double normalCdfComplement(double x) {
		double t = 1.0 / (1.0 + 0.2316419 * x);
		double poly = t * (0.319381530 + t * (-0.356563782
				+ t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
		return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI) * poly;
	}

    private double[] collectBinary(Dataset<Row> df, String posThird, String colName) {
        List<Row> rows = df.filter(col("pos_third").equalTo(posThird))
                           .select(colName).collectAsList();
        double[] arr = new double[rows.size()];
        for (int i = 0; i < rows.size(); i++) arr[i] = ((Number) rows.get(i).get(0)).intValue();
        return arr;
    }
}
