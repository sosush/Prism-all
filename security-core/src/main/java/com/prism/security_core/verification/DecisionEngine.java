package com.prism.security_core.verification;


public class DecisionEngine {
    public static double calculateScore(VerificationRequest r) {
        return 0.30 * r.getEyeScore()
                + 0.25 * r.getSkinScore()
                + 0.25 * r.getPulseScore()
                + 0.20 * r.getFlashScore();
    }
}
