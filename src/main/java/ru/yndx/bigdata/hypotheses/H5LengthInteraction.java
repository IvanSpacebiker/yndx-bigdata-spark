package ru.yndx.bigdata.hypotheses;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.yndx.bigdata.pipeline.Datasets;

import static org.apache.spark.sql.functions.*;

/**
 * H5 — Listen+ probability drops faster with track length for recommended tracks.
 *
 * Method:
 *   1. Bucket analysis: Listen+ rate by length_bucket × is_organic
 *   2. Logistic regression: Listen+ ~ is_organic + track_length + is_organic×track_length
 *      Positive interaction coefficient → recs lose Listen+ faster with length
 */
public class H5LengthInteraction {

    private static final Logger log = LoggerFactory.getLogger(H5LengthInteraction.class);
    private static final long SEED = 42L;
    private static final int SAMPLE_SIZE = 100_000;

    private final SparkSession spark;
    private final Datasets ds;

    public H5LengthInteraction(SparkSession spark, Datasets ds) {
        this.spark = spark;
        this.ds    = ds;
    }

    public HypothesisResult run() {
        log.info("=== H5: Listen+ vs track length interaction ===");

		Dataset<Row> df = ds.listensEnriched
				.filter(col("played_ratio_pct").isNotNull().and(col("track_length_seconds").isNotNull()))
				.select("uid", "is_organic", "played_ratio_pct", "track_length_seconds", "listen_plus")
				.cache();

        // ── 1. Bucket analysis ────────────────────────────────────────
        Dataset<Row> bucketed = df.withColumn("len_bucket",
                when(col("track_length_seconds").lt(90),  "<1.5m")
               .when(col("track_length_seconds").lt(150), "1.5-2.5m")
               .when(col("track_length_seconds").lt(210), "2.5-3.5m")
               .when(col("track_length_seconds").lt(270), "3.5-4.5m")
               .when(col("track_length_seconds").lt(360), "4.5-6m")
               .otherwise("6m+"));

        log.info("Listen+ rate by length_bucket × is_organic:");
        bucketed.groupBy("len_bucket", "is_organic")
                .agg(avg("listen_plus").alias("lp_rate"), count("*").alias("n"))
                .orderBy("len_bucket", "is_organic")
                .show(50);

        // ── 2. Logistic Regression with interaction term ───────────────
		Dataset<Row> sample = df
				.withColumn("is_organic", col("is_organic").cast("double"))
				.sample(false, computeFraction(df, SAMPLE_SIZE), SEED)
				.limit(SAMPLE_SIZE)
				.cache();

// Manual standardization: compute mean & std per column
		Row stats = sample.select(
				mean("is_organic").alias("org_mean"),
				stddev("is_organic").alias("org_std"),
				mean(col("track_length_seconds").cast("double")).alias("len_mean"),
				stddev(col("track_length_seconds").cast("double")).alias("len_std")
		).first();

		double orgMean = stats.getDouble(0), orgStd  = stats.getDouble(1);
		double lenMean = stats.getDouble(2), lenStd  = stats.getDouble(3);

// z-score columns, then build interaction term — no VectorUDT involved
		Dataset<Row> withScaledOrg = sample
				.withColumn("s_organic",
						col("is_organic").minus(orgMean).divide(orgStd))
				.withColumn("s_length",
						col("track_length_seconds").cast("double").minus(lenMean).divide(lenStd))
				.withColumn("interaction",
						col("s_organic").multiply(col("s_length")));

		VectorAssembler finalAssembler = new VectorAssembler()
				.setInputCols(new String[]{"s_organic", "s_length", "interaction"})
				.setOutputCol("features");

		Dataset<Row> modelData = finalAssembler.transform(withScaledOrg)
				.withColumn("label", col("listen_plus").cast("double"));

		LogisticRegression lr = new LogisticRegression()
				.setMaxIter(300)
				.setFeaturesCol("features")
				.setLabelCol("label");

		LogisticRegressionModel model = lr.fit(modelData);
		double[] coefs = model.coefficients().toArray();

		log.info("Logistic Regression coefficients:");
		log.info("  is_organic               : {:+.4f}", coefs[0]);
		log.info("  track_length             : {:+.4f}", coefs[1]);
		log.info("  is_organic × track_length: {:+.4f}", coefs[2]);

        double interactionCoef = coefs[2];
        // Positive interaction → organic loses Listen+ faster with length → rec drops faster
        // Actually: if organic = 1, interaction ↑ → stronger negative of length for organic
        // So: coef[interaction] > 0 means recs benefit more from short tracks → H5 confirmed
        boolean confirmed = interactionCoef > 0;

        String details = String.format(
                "LogReg interaction (is_organic×length) = %+.4f %s",
                interactionCoef,
                confirmed ? "(>0 → рекомендательные теряют Listen+ быстрее с ростом длины)"
                          : "(<0 → органические теряют Listen+ быстрее)");

        return new HypothesisResult("H5",
                "Listen+ снижается быстрее с длиной для рекомендаций",
                confirmed, details);
    }

    private double computeFraction(Dataset<Row> df, int target) {
        long n = df.count();
        return n <= target ? 1.0 : (double) target / n * 1.2;
    }
}
