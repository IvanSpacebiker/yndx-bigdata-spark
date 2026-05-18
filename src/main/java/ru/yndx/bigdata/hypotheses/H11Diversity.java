package ru.yndx.bigdata.hypotheses;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.yndx.bigdata.pipeline.Datasets;
import ru.yndx.bigdata.stats.StatUtils;

import java.util.*;
import java.util.stream.Collectors;

import static org.apache.spark.sql.functions.*;

/**
 * H11 — In sliding windows of 100 events, windows where >50% of tracks
 *        are recommended have higher Shannon entropy (item_id diversity)
 *        and higher intra-list cosine diversity.
 *
 * Method:
 *   1. Sample 1000 users
 *   2. Sliding windows (size=100, step=50) over each user's listen history
 *   3. Per window: compute rec_share, Shannon entropy of item_ids, cosine diversity
 *   4. Mann-Whitney: rec-dominated vs org-dominated windows
 */
public class H11Diversity {

    private static final Logger log = LoggerFactory.getLogger(H11Diversity.class);
    private static final int WINDOW_SIZE = 100;
    private static final int STEP        = 50;
    private static final int N_USERS     = 1000;
    private static final int MAX_VECS    = 30;   // cap for cosine diversity speed
    private static final long SEED       = 42L;

    private final SparkSession spark;
    private final Datasets ds;

    public H11Diversity(SparkSession spark, Datasets ds) {
        this.spark = spark;
        this.ds    = ds;
    }

    public HypothesisResult run() {
        log.info("=== H11: Diversity in recommendation-dominated windows ===");

        // 1. Sample users
        List<Long> sampleUids = ds.listensEnriched
                .select("uid").distinct()
                .sample(false, 0.01, SEED)
                .limit(N_USERS)
                .select("uid")
                .collectAsList()
                .stream()
                .map(r -> ((Number) r.get(0)).longValue())
                .collect(Collectors.toList());

        // 2. Collect listen history
        List<Row> userListens = ds.listensEnriched
                .filter(col("uid").isin(sampleUids.toArray()))
                .select("uid", "item_id", "is_organic", "abs_time_sec")
                .orderBy("uid", "abs_time_sec")
                .collectAsList();

        // 3. Try to load embeddings (optional)
        String embPath = ds.listensEnriched.sparkSession()
                           .conf().get("spark.app.name");  // placeholder
        // We'll skip cosine diversity if embeddings aren't available
        Map<Long, float[]> embDict = new HashMap<>();  // empty = no cosine diversity

        // 4. Compute windows
        List<double[]> windows = computeWindows(userListens, embDict);
        log.info("Windows computed: {}", windows.size());

        if (windows.isEmpty()) {
            return new HypothesisResult("H11",
                    "Рекомендации увеличивают разнообразие",
                    false, "No windows computed");
        }

        // 5. Split into rec-dominated vs org-dominated
        List<Double> recEntropy = new ArrayList<>(), orgEntropy = new ArrayList<>();
        List<Double> recDiv     = new ArrayList<>(), orgDiv     = new ArrayList<>();

        for (double[] w : windows) {
            // w = [rec_share, entropy, diversity]
            if (w[0] > 0.5) {
                recEntropy.add(w[1]);
                if (!Double.isNaN(w[2])) recDiv.add(w[2]);
            } else {
                orgEntropy.add(w[1]);
                if (!Double.isNaN(w[2])) orgDiv.add(w[2]);
            }
        }

        double[] recEnt = toArray(recEntropy);
        double[] orgEnt = toArray(orgEntropy);

        log.info("Entropy: rec_mean={:.4f}, org_mean={:.4f}",
                StatUtils.mean(recEnt), StatUtils.mean(orgEnt));

        double[] mwEnt = StatUtils.mannWhitneyTwoSided(recEnt, orgEnt);
        double pEnt = mwEnt[1];

        String divDetails = "";
        double pDiv = Double.NaN;
        if (!recDiv.isEmpty() && !orgDiv.isEmpty()) {
            double[] mwDiv = StatUtils.mannWhitneyTwoSided(toArray(recDiv), toArray(orgDiv));
            pDiv = mwDiv[1];
            divDetails = String.format(", Diversity p=%.4e", pDiv);
        }

        log.info("Mann-Whitney entropy p={}", pEnt);

        boolean confirmed = pEnt < 0.05 && StatUtils.mean(recEnt) > StatUtils.mean(orgEnt);
        String details = String.format(
                "Entropy: rec=%.3f, org=%.3f, p=%.4e%s",
                StatUtils.mean(recEnt), StatUtils.mean(orgEnt), pEnt, divDetails);

        return new HypothesisResult("H11",
                "Рекомендации увеличивают разнообразие потребления",
                confirmed, details);
    }

    private List<double[]> computeWindows(List<Row> userListens,
                                           Map<Long, float[]> embDict) {
        List<double[]> result = new ArrayList<>();

        // Group by uid
        Map<Long, List<long[]>> byUser = new LinkedHashMap<>();
        for (Row r : userListens) {
            long uid = ((Number) r.get(0)).longValue();
            long iid = ((Number) r.get(1)).longValue();
            int  org = ((Number) r.get(2)).intValue();
            byUser.computeIfAbsent(uid, k -> new ArrayList<>())
                  .add(new long[]{iid, org});
        }

        for (List<long[]> events : byUser.values()) {
            if (events.size() < WINDOW_SIZE) continue;

            for (int start = 0; start + WINDOW_SIZE <= events.size(); start += STEP) {
                List<long[]> window = events.subList(start, start + WINDOW_SIZE);

                double recShare = window.stream()
                        .mapToInt(e -> (int) e[1])
                        .filter(o -> o == 0)
                        .count() / (double) window.size();

                double entropy = shannonEntropy(window);

                double diversity = Double.NaN;
                if (!embDict.isEmpty()) {
                    diversity = cosineDiversity(window, embDict);
                }

                result.add(new double[]{recShare, entropy, diversity});
            }
        }
        return result;
    }

    /** Shannon entropy of item_id distribution within the window (bits). */
    private double shannonEntropy(List<long[]> window) {
        Map<Long, Integer> counts = new HashMap<>();
        for (long[] e : window) counts.merge(e[0], 1, Integer::sum);
        double entropy = 0;
        int n = window.size();
        for (int c : counts.values()) {
            double p = c / (double) n;
            entropy -= p * (Math.log(p) / Math.log(2));
        }
        return entropy;
    }

    /** Average pairwise cosine distance (1 - similarity) — capped at MAX_VECS. */
    private double cosineDiversity(List<long[]> window, Map<Long, float[]> embDict) {
        List<float[]> vecs = new ArrayList<>();
        for (long[] e : window) {
            float[] v = embDict.get(e[0]);
            if (v != null) vecs.add(v);
            if (vecs.size() >= MAX_VECS) break;
        }
        if (vecs.size() < 2) return Double.NaN;

        double sum = 0;
        int cnt = 0;
        for (int i = 0; i < vecs.size(); i++) {
            for (int j = i + 1; j < vecs.size(); j++) {
                sum += 1.0 - cosineSim(vecs.get(i), vecs.get(j));
                cnt++;
            }
        }
        return cnt > 0 ? sum / cnt : Double.NaN;
    }

    private double cosineSim(float[] a, float[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
        }
        double d = Math.sqrt(na) * Math.sqrt(nb);
        return d < 1e-9 ? 0 : dot / d;
    }

    private double[] toArray(List<Double> list) {
        return list.stream().mapToDouble(Double::doubleValue).toArray();
    }
}
