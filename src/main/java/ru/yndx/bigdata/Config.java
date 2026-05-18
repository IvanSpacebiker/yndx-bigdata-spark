package ru.yndx.bigdata;

public class Config {

    public final String dataDir;
    public final String master;
    public final long   randomSeed;
    public final int    sampleSize;   // per-group cap for expensive tests

    private Config(String dataDir, String master, long randomSeed, int sampleSize) {
        this.dataDir    = dataDir;
        this.master     = master;
        this.randomSeed = randomSeed;
        this.sampleSize = sampleSize;
    }

    public static Config parse(String[] args) {
        String dataDir    = "data";
        String master     = "local[*]";
        long   seed       = 42L;
        int    sampleSize = 500_000;

        for (int i = 0; i < args.length - 1; i++) {
            switch (args[i]) {
                case "--data-dir":    dataDir    = args[++i]; break;
                case "--master":      master     = args[++i]; break;
                case "--seed":        seed       = Long.parseLong(args[++i]); break;
                case "--sample-size": sampleSize = Integer.parseInt(args[++i]); break;
            }
        }

        return new Config(dataDir, master, seed, sampleSize);
    }
}
