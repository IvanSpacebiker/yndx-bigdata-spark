package ru.yndx.bigdata.hypotheses;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.yndx.bigdata.pipeline.Datasets;
import ru.yndx.bigdata.stats.StatUtils;

import java.util.List;

import static org.apache.spark.sql.functions.*;

/**
 * H8 — In the top activity quartile (Q4), Listen+ rate and like-rate
 *       for recommended tracks (is_organic=0) are higher than in Q1.
 *
 * Method:
 *   1. Compute per-user total_listens → activity_quartile (Q1..Q4)
 *   2. Filter to recommended listens only
 *   3. Compute Listen+ rate and like-rate per quartile
 *   4. Spearman ρ between quartile rank and Listen+ (trend test)
 */
public class H8ExperiencedUsers {

    private static final Logger log = LoggerFactory.getLogger(H8ExperiencedUsers.class);
    private static final long SEED = 42L;

    private final SparkSession spark;
    private final Datasets ds;

    public H8ExperiencedUsers(SparkSession spark, Datasets ds) {
        this.spark = spark;
        this.ds    = ds;
    }

    public HypothesisResult run() {
        log.info("=== H8: Recommendation effectiveness by user experience quartile ===");

        // 1. User activity quartile
        Dataset<Row> userActivity = ds.listensEnriched
                .groupBy("uid")
                .agg(count("*").alias("total_listens"))
                .withColumn("activity_quartile",
                        ntile(4).over(
                                org.apache.spark.sql.expressions.Window
                                        .orderBy("total_listens")
                        )
                );
        // ntile(4): 1=least active, 4=most active

        // 2. Join back to listens, filter to recommended
        Dataset<Row> recListens = ds.listensEnriched
                .filter(col("is_organic").equalTo(0))
                .join(userActivity.select("uid", "activity_quartile"), "uid");

        log.info("Listen+ rate for recommended tracks by activity quartile:");
        recListens.groupBy("activity_quartile")
                  .agg(
                      avg("listen_plus").alias("lp_rate"),
                      count("*").alias("n")
                  )
                  .orderBy("activity_quartile")
                  .show();

        // 3. Like-rate for recommended tracks
        Dataset<Row> likeFlag = ds.likes
                .select("uid", "item_id").distinct()
                .withColumn("liked", lit(1));

        Dataset<Row> recAgg = recListens
                .groupBy("uid", "item_id", "activity_quartile")
                .agg(count("*").alias("n"))
                .join(likeFlag, new String[] {"uid", "item_id"}, "left")
                .withColumn("liked", coalesce(col("liked"), lit(0)));

        log.info("Like rate for recommended tracks by activity quartile:");
        recAgg.groupBy("activity_quartile")
              .agg(avg("liked").alias("like_rate"), count("*").alias("pairs"))
              .orderBy("activity_quartile")
              .show();

        // 4. Trend test: Spearman ρ(quartile, listen_plus) on sample
        List<Row> rows = recListens.select("activity_quartile", "listen_plus")
                                   .sample(false, 0.1, SEED)
                                   .limit(500_000)
                                   .collectAsList();

        double[] quartiles  = new double[rows.size()];
        double[] listenPlus = new double[rows.size()];
        for (int i = 0; i < rows.size(); i++) {
            quartiles[i]  = ((Number) rows.get(i).get(0)).intValue();
            listenPlus[i] = ((Number) rows.get(i).get(1)).intValue();
        }

        double[] spearman = StatUtils.spearman(quartiles, listenPlus);
        log.info("Trend Spearman ρ={}, p={}", spearman[0], spearman[1]);

        boolean confirmed = spearman[1] < 0.05 && spearman[0] > 0;
        String details = String.format(
                "Trend Spearman ρ=%.4f, p=%.4e (positive ρ → recs improve with experience)",
                spearman[0], spearman[1]);

        return new HypothesisResult("H8",
                "Опытные пользователи: рекомендации эффективнее",
                confirmed, details);
    }
}
