# C3 K1 Covariance-Drift Gate

Status: `P8_NOT_PROMOTED_NO_TRUE_COVARIANCE`.

This gate inherits `artifacts/covariance_headroom/`: only compact magnitude and mean summaries are cached, so centered diagonal/off-diagonal covariance cannot be decomposed. Since tight clip is now closed as a universal method, this proxy no longer promotes scale/clip GPU work. P8 online scale-refresh requires a real covariance cache or a tiny capture targeted at that question.
