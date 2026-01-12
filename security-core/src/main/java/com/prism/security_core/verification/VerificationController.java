// package com.prism.security_core.verification;

// import java.util.Map;

// import org.springframework.beans.factory.annotation.Autowired;
// import org.springframework.beans.factory.annotation.Value;
// import org.springframework.http.ResponseEntity;
// import org.springframework.security.core.Authentication;
// import org.springframework.web.bind.annotation.PostMapping;
// import org.springframework.web.bind.annotation.RequestBody;
// import org.springframework.web.bind.annotation.RequestHeader;
// import org.springframework.web.bind.annotation.RequestMapping;
// import org.springframework.web.bind.annotation.RestController;

// import com.prism.security_core.blockchain.BlockchainService;

// @RestController
// @RequestMapping("/api/verify")
// public class VerificationController {

//     @Value("${python.api.key}")
//     private String pythonApiKey;

//     @Autowired
// private BlockchainService blockchainService;


//     @PostMapping("/human")
//     public ResponseEntity<?> verifyHuman(
//             @RequestBody VerificationRequest request,
//             @RequestHeader(value = "X-API-KEY", required = false) String apiKey, Authentication auth) {


//         String username = auth.getName();        
//         // 1️⃣ API key check
//         if (apiKey == null) {
//             return ResponseEntity.badRequest().body("X-API-KEY missing");
//         }

//         if (!apiKey.equals(pythonApiKey)) {
//             return ResponseEntity.status(401).body("Unauthorized");
//         }

//         // 2️⃣ Score validation
//         if (!validScores(request)) {
//             return ResponseEntity.badRequest().body("Invalid score values");
//         }

//         // 3️⃣ Decision engine
//         double score = DecisionEngine.calculateScore(request);

//         if (score < 0.75) {
//             return ResponseEntity.status(403).body("Verification Failed");
//         }

//         // 4️⃣ SUCCESS RESPONSE (Blockchain paused)
//         return ResponseEntity.ok(
//                 Map.of(
//                         "status", "VERIFIED",
//                         "confidenceScore", score,
//                         "note", "Blockchain integration disabled for demo"));
//     }

//     private boolean validScores(VerificationRequest r) {
//         return r.getEyeScore() >= 0 && r.getEyeScore() <= 1
//                 && r.getSkinScore() >= 0 && r.getSkinScore() <= 1
//                 && r.getPulseScore() >= 0 && r.getPulseScore() <= 1
//                 && r.getFlashScore() >= 0 && r.getFlashScore() <= 1;
//     }
// }


package com.prism.security_core.verification;

import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.prism.security_core.blockchain.BlockchainService;

@RestController
@RequestMapping("/api/verify")
public class VerificationController {

    @Value("${python.api.key}")
    private String pythonApiKey;

    @Autowired
    private BlockchainService blockchainService;

    @PostMapping("/human")
    public ResponseEntity<?> verifyHuman(
            @RequestBody VerificationRequest request,
            @RequestHeader(value = "X-API-KEY", required = false) String apiKey,
            Authentication auth) {

        String username = auth.getName();

        // 1️⃣ API key check
        if (apiKey == null) {
            return ResponseEntity.badRequest().body("X-API-KEY missing");
        }

        if (!apiKey.equals(pythonApiKey)) {
            return ResponseEntity.status(401).body("Unauthorized");
        }

        // 2️⃣ Score validation
        if (!validScores(request)) {
            return ResponseEntity.badRequest().body("Invalid score values");
        }

        // 3️⃣ Decision engine
        double score = DecisionEngine.calculateScore(request);

        if (score < 0.75) {
            return ResponseEntity.status(403).body("Verification Failed");
        }

        // 4️⃣ SUCCESS → BLOCKCHAIN WRITE
        try {
            String proofData = username + ":" + score + ":" + System.currentTimeMillis();
            String txHash = blockchainService.storeProof(proofData);

            return ResponseEntity.ok(
                    Map.of(
                            "status", "VERIFIED",
                            "confidenceScore", score,
                            "blockchainTx", txHash));
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.ok(
                    Map.of(
                            "status", "VERIFIED",
                            "confidenceScore", score,
                            "blockchainTx", "FAILED_TO_WRITE"));
        }
    }

    private boolean validScores(VerificationRequest r) {
        return r.getEyeScore() >= 0 && r.getEyeScore() <= 1
                && r.getSkinScore() >= 0 && r.getSkinScore() <= 1
                && r.getPulseScore() >= 0 && r.getPulseScore() <= 1
                && r.getFlashScore() >= 0 && r.getFlashScore() <= 1;
    }
}
