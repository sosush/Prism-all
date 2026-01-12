package com.prism.security_core.verification;

import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
public class VerificationRequest {

    private String sessionId;
    private Double eyeScore;
    private Double skinScore;
    private Double pulseScore;
    private Double flashScore;
    private String wallet;
}
