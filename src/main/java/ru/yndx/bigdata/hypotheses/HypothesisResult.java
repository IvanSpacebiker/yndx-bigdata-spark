package ru.yndx.bigdata.hypotheses;

/** Immutable result returned by every hypothesis class. */
public final class HypothesisResult {

    public final String  id;           // "H1", "H2", …
    public final String  description;  // short description
    public final boolean confirmed;    // p < 0.05 in the correct direction
    public final String  details;      // stats summary line

    public HypothesisResult(String id, String description,
                            boolean confirmed, String details) {
        this.id          = id;
        this.description = description;
        this.confirmed   = confirmed;
        this.details     = details;
    }

    @Override
    public String toString() {
        return String.format("%s [%s] %s | %s",
                id, confirmed ? "✅" : "❌", description, details);
    }
}
