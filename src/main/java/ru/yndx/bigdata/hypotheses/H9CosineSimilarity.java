package ru.yndx.bigdata.hypotheses;

import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.yndx.bigdata.pipeline.Datasets;
import ru.yndx.bigdata.stats.StatUtils;
import scala.collection.mutable.WrappedArray;

import java.util.*;

import static org.apache.spark.sql.functions.*;

/**
 * H9 — Recommended tracks with higher cosine similarity to the user's
 *       recent organic listening history are more likely to be Listen+.
 *
 * Method (mirrors notebook):
 *   1. Read embeddings.parquet (only item_ids needed for the user sample)
 *   2. For each user (sample of 2000), iterate events in time order:
 *      - If is_organic=1 → append to rolling window
 *      - If is_organic=0 and window ≥ 3 → compute cosine sim(rec, mean(window))
 *   3. Spearman ρ(cosine_sim, listen_plus)
 *
 * Note: embeddings.parquet must exist at {dataDir}/embeddings.parquet.
 *       Skips gracefully if the file is absent.
 */
public class H9CosineSimilarity {

    private static final Logger log = LoggerFactory.getLogger(H9CosineSimilarity.class);
    private static final int WINDOW = 20;
    private static final int N_USERS = 2000;
    private static final int MIN_ORGANIC = 5;
    private static final long SEED = 42L;

    private final SparkSession spark;
    private final Datasets ds;
    private final String dataDir;

    public H9CosineSimilarity(SparkSession spark, Datasets ds, String dataDir) {
        this.spark   = spark;
        this.ds      = ds;
        this.dataDir = dataDir;
    }

    public HypothesisResult run() {
        log.info("=== H9: Cosine similarity to organic history ===");

        String embPath = dataDir + "/embeddings.parquet";
        try {
            spark.read().parquet(embPath); // check exists
        } catch (Exception e) {
            log.warn("embeddings.parquet not found at {}. Skipping H9.", embPath);
            return new HypothesisResult("H9",
                    "Аудио-похожесть → лучше дослушивается",
                    false, "SKIPPED — embeddings.parquet not available");
        }

        // 1. Sample 2000 users with enough organic + at least 1 rec listen
        List<Long> sampleUids = sampleEligibleUsers();
        log.info("Users for H9 analysis: {}", sampleUids.size());

        if (sampleUids.isEmpty()) {
            return new HypothesisResult("H9",
                    "Аудио-похожесть → лучше дослушивается",
                    false, "No eligible users found");
        }

        // 2. Collect needed item_ids
        Set<Long> neededItems = new HashSet<>();
        List<Row> userListens = ds.listensEnriched
                .filter(col("uid").isin(sampleUids.toArray()))
                .select("uid", "item_id", "abs_time_sec", "is_organic", "listen_plus")
                .orderBy("uid", "abs_time_sec")
                .collectAsList();

        for (Row r : userListens) neededItems.add(((Number) r.get(1)).longValue());
        log.info("Needed embeddings: {}", neededItems.size());

        // 3. Load embeddings for needed items only
        Map<Long, float[]> embDict = loadEmbeddings(embPath, neededItems);
        log.info("Loaded embeddings: {}", embDict.size());

        // 4. Compute cosine similarities per user
        List<double[]> records = new ArrayList<>();  // [cosine_sim, listen_plus]
        processUsers(userListens, embDict, records);

        if (records.isEmpty()) {
            return new HypothesisResult("H9",
                    "Аудио-похожесть → лучше дослушивается",
                    false, "No similarity records computed");
        }

        double[] sims = new double[records.size()];
        double[] lps  = new double[records.size()];
        for (int i = 0; i < records.size(); i++) {
            sims[i] = records.get(i)[0];
            lps[i]  = records.get(i)[1];
        }

        // 5. Spearman rho
        double[] spearman = StatUtils.spearman(sims, lps);
        log.info("Spearman ρ(cosine_sim, listen_plus)={}, p={}", spearman[0], spearman[1]);

        boolean confirmed = spearman[1] < 0.05 && spearman[0] > 0;
        String details = String.format(
                "Spearman ρ=%.4f, p=%.4e (n=%d events)", spearman[0], spearman[1], records.size());

        return new HypothesisResult("H9",
                "Аудио-похожесть → лучше дослушивается",
                confirmed, details);
    }

    private List<Long> sampleEligibleUsers() {
        List<Row> rows = ds.listensEnriched
                .groupBy("uid")
                .agg(
                    sum(when(col("is_organic").equalTo(1), 1).otherwise(0)).alias("organic_cnt"),
                    sum(when(col("is_organic").equalTo(0), 1).otherwise(0)).alias("rec_cnt")
                )
                .filter(col("organic_cnt").geq(MIN_ORGANIC).and(col("rec_cnt").geq(1)))
                .select("uid")
                .sample(false, 0.01, SEED)
                .limit(N_USERS)
                .collectAsList();

        List<Long> uids = new ArrayList<>();
        for (Row r : rows) uids.add(((Number) r.get(0)).longValue());
        return uids;
    }

    private Map<Long, float[]> loadEmbeddings(String path, Set<Long> neededItems) {
        // Read in batches; filter to needed items
        List<Row> rows = spark.read().parquet(path)
                .filter(col("item_id").isin(neededItems.toArray()))
                .select("item_id", "normalized_embed")
                .collectAsList();

        Map<Long, float[]> dict = new HashMap<>();
        for (Row r : rows) {
            long itemId = ((Number) r.get(0)).longValue();
            WrappedArray<?> raw = (WrappedArray<?>) r.get(1);
            float[] vec = new float[raw.size()];
            for (int i = 0; i < vec.length; i++) vec[i] = ((Number) raw.apply(i)).floatValue();
            dict.put(itemId, vec);
        }
        return dict;
    }

    private void processUsers(List<Row> userListens, Map<Long, float[]> embDict,
                               List<double[]> out) {
        long curUid = Long.MIN_VALUE;
        Deque<long[]> organicHistory = new ArrayDeque<>();  // [item_id]
        // Actually store as item_ids only; look up embedding on demand
        Deque<Long> orgItemIds = new ArrayDeque<>();

        for (Row row : userListens) {
            long uid      = ((Number) row.get(0)).longValue();
            long itemId   = ((Number) row.get(1)).longValue();
            int  isOrg    = ((Number) row.get(3)).intValue();
            int  lp       = ((Number) row.get(4)).intValue();

            if (uid != curUid) {
                curUid = uid;
                orgItemIds.clear();
            }

            if (isOrg == 1) {
                orgItemIds.addLast(itemId);
                if (orgItemIds.size() > WINDOW) orgItemIds.pollFirst();
            } else if (orgItemIds.size() >= 3) {
                float[] recVec = embDict.get(itemId);
                if (recVec == null) continue;

                // Mean of organic window
                float[] mean = new float[recVec.length];
                int count = 0;
                for (long oid : orgItemIds) {
                    float[] ov = embDict.get(oid);
                    if (ov == null) continue;
                    for (int i = 0; i < mean.length; i++) mean[i] += ov[i];
                    count++;
                }
                if (count == 0) continue;
                for (int i = 0; i < mean.length; i++) mean[i] /= count;

                double sim = cosineSim(recVec, mean);
                out.add(new double[]{sim, lp});
            }
        }
    }

    private double cosineSim(float[] a, float[] b) {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot   += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        double denom = Math.sqrt(normA) * Math.sqrt(normB);
        return denom < 1e-9 ? 0 : dot / denom;
    }
}
