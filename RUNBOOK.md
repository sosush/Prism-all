# PRISM Protocol Runbook

Complete deployment, configuration, and operations guide for the PRISM Protocol verification system.

---

## Table of Contents

1.  [Overview](#overview)
2.  [System Requirements](#system-requirements)
3.  [Architecture Overview](#architecture-overview)
4.  [Quick Start](#quick-start)
5.  [Component Setup](#component-setup)
    - [Python Backend (Vision Engine)](#1-python-backend-vision-engine)
    - [Java Backend (Security Core)](#2-java-backend-security-core)
    - [Smart Contracts](#3-smart-contracts)
    - [Frontend](#4-frontend)
6.  [Configuration Reference](#configuration-reference)
7.  [API Reference](#api-reference)
8.  [On-Chain Verification](#on-chain-verification)
9.  [Troubleshooting](#troubleshooting)
10. [Security Considerations](#security-considerations)

---

## Overview

PRISM Protocol consists of four primary components:

| Component | Technology | Purpose |
|:--|:--|:--|
| **Vision Backend** | Python / FastAPI | Processes video frames, runs physics engines (corneal, SSS, rPPG), generates liveness scores |
| **Security Core** | Java / Spring Boot | Handles authentication, orchestrates blockchain writes, manages attestations |
| **Smart Contracts** | Solidity / Foundry | On-chain registry, proof verification, Soulbound Token minting |
| **Frontend** | Next.js / React | User interface, webcam capture, Chroma Challenge execution |

This runbook covers deployment and operation of each component.

---

## System Requirements

### Development Environment

| Requirement | Minimum | Recommended |
|:--|:--|:--|
| **Operating System** | Linux / macOS | Ubuntu 22.04 LTS |
| **Python** | 3.10 | 3.11 |
| **Java** | 17 | 21+ |
| **Node.js** | 18 | 20 LTS |
| **RAM** | 8 GB | 16 GB |
| **GPU** | Not required | NVIDIA GPU (CUDA) for faster inference |

### External Dependencies

| Service | Purpose | Required |
|:--|:--|:--|
| **MariaDB / MySQL** | Attestation persistence | Yes (for Security Core) |
| **Ethereum RPC** | Blockchain interaction | Yes (Infura, Alchemy, or self-hosted) |
| **Funded Wallet** | Gas for on-chain transactions | Yes (Sepolia ETH for testnet) |

### Software Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# Key packages:
# - fastapi, uvicorn (API server)
# - opencv-python, mediapipe (computer vision)
# - torch, numpy, scipy (ML/signal processing)
# - websockets (real-time streaming)

# Java dependencies (managed via Maven)
# - Spring Boot 3.x
# - Web3j (Ethereum interaction)
# - MariaDB connector

# Blockchain tools
# - Foundry (forge, cast, anvil)
curl -L https://foundry.paradigm.xyz | bash
foundryup
```

---

## Architecture Overview

```
                                    PRISM PROTOCOL DEPLOYMENT
+-------------------------------------------------------------------------------------------+
|                                                                                           |
|   USER                                                                                    |
|     |                                                                                     |
|     v                                                                                     |
|   +-------------------+                                                                   |
|   |    FRONTEND       |  Port 3000                                                        |
|   |    (Next.js)      |------------------------------------------------------------+      |
|   +-------------------+                                                            |      |
|           |                                                                        |      |
|           | WebSocket (frames + challenge metadata)                                |      |
|           v                                                                        |      |
|   +-------------------+                                                            |      |
|   |  VISION BACKEND   |  Port 8000                                                 |      |
|   |  (Python/FastAPI) |                                                            |      |
|   +-------------------+                                                            |      |
|           |                                                                        |      |
|           | HTTP POST /api/verify/human                                            |      |
|           | (wallet, sessionId, eyeScore, skinScore, pulseScore, flashScore)       |      |
|           v                                                                        |      |
|   +-------------------+        +-------------------+        +-------------------+  |      |
|   |  SECURITY CORE    |  --->  |    MariaDB        |        |   BLOCKCHAIN      |  |      |
|   |  (Java/Spring)    |        |    (Persistence)  |        |   (Sepolia/Base)  |  |      |
|   |  Port 8080        |        +-------------------+        +-------------------+  |      |
|   +-------------------+                                              ^             |      |
|           |                                                          |             |      |
|           +----------------------------------------------------------+             |      |
|             Web3j: mintAttestation(wallet, proofHash, expiry)                      |      |
|                                                                                    |      |
|           +-----------------------------------------------------------------------|------+
|           | Return: { status, blockchainTx, tokenId, expiresAt }                  |
|           v                                                                        |
|       FRONTEND displays success + transaction hash                                 |
|                                                                                           |
+-------------------------------------------------------------------------------------------+
```

---

## Quick Start

For local development with all components:

```bash
# Terminal 1: Start MariaDB (if not running)
sudo systemctl start mariadb

# Terminal 2: Start Python Vision Backend
cd prism-protocol
pip install -r requirements.txt
python app/server.py
# Runs on http://localhost:8000

# Terminal 3: Start Java Security Core
export SPRING_DATASOURCE_URL="jdbc:mysql://localhost:3306/prism_db"
export SPRING_DATASOURCE_USERNAME="prism"
export SPRING_DATASOURCE_PASSWORD="your_password"
export WEB3_RPC_URL="https://sepolia.infura.io/v3/YOUR_PROJECT_ID"
export WEB3_CONTRACT_ADDRESS="0xD96fF6ba63E803C877c82999C0C7DD829C3415B8"
export WEB3_WALLET_PRIVATEKEY="0xYOUR_PRIVATE_KEY"

./security-core/mvnw -f security-core/pom.xml spring-boot:run
# Runs on http://localhost:8080

# Terminal 4: Start Frontend (if available)
cd frontend
npm install
npm run dev
# Runs on http://localhost:3000
```

---

## Component Setup

### 1. Python Backend (Vision Engine)

The Vision Backend processes video frames and runs the four detection engines.

#### Installation

```bash
cd prism-protocol

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Running the Server

```bash
# Development mode
python app/server.py

# Production mode with Uvicorn
uvicorn app.server:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Key Files

| File | Purpose |
|:--|:--|
| `app/server.py` | FastAPI application entry point |
| `app/main.py` | Core liveness detection logic |
| `app/sanity_check.py` | Physics engine validation |
| `app/eval.py` | Model evaluation utilities |

#### Environment Variables

| Variable | Default | Description |
|:--|:--|:--|
| `PRISM_HOST` | `0.0.0.0` | Server bind address |
| `PRISM_PORT` | `8000` | Server port |
| `SECURITY_CORE_URL` | `http://localhost:8080` | Java backend URL |
| `SECURITY_CORE_API_KEY` | `prism-python-secret` | API key for Security Core |

---

### 2. Java Backend (Security Core)

The Security Core handles authentication, attestation management, and blockchain interactions.

#### Prerequisites

```bash
# Ensure Java is installed
java --version  # Should be 17+

# Ensure MariaDB is running
sudo systemctl status mariadb
```

#### Database Setup

```sql
-- Connect to MariaDB
mysql -u root -p

-- Create database and user
CREATE DATABASE IF NOT EXISTS prism_db;
CREATE USER IF NOT EXISTS 'prism'@'localhost' IDENTIFIED BY 'choose_a_secure_password';
GRANT ALL PRIVILEGES ON prism_db.* TO 'prism'@'localhost';
FLUSH PRIVILEGES;
```

#### Environment Variables (Required)

```bash
# Database Configuration
export SPRING_DATASOURCE_URL="jdbc:mysql://localhost:3306/prism_db"
export SPRING_DATASOURCE_USERNAME="prism"
export SPRING_DATASOURCE_PASSWORD="your_password"

# Blockchain Configuration
export WEB3_RPC_URL="https://sepolia.infura.io/v3/YOUR_PROJECT_ID"
export WEB3_CONTRACT_ADDRESS="0xD96fF6ba63E803C877c82999C0C7DD829C3415B8"
export WEB3_WALLET_PRIVATEKEY="0xYOUR_MINTER_PRIVATE_KEY"

# Optional
export PRISM_ATTESTATION_TTLSECONDS=604800  # 7 days default
```

#### Running the Server

```bash
cd prism-protocol

# Using Maven Wrapper
./security-core/mvnw -f security-core/pom.xml spring-boot:run

# Or build and run JAR
./security-core/mvnw -f security-core/pom.xml package -DskipTests
java -jar security-core/target/security-core-0.0.1-SNAPSHOT.jar
```

#### Key Files

| File | Purpose |
|:--|:--|
| `security-core/src/.../VerificationController.java` | API endpoint handler |
| `security-core/src/.../BlockchainService.java` | Web3j blockchain interaction |
| `security-core/src/.../PrismVerificationContract.java` | Smart contract wrapper |

---

### 3. Smart Contracts

The `PRISMRegistry` contract is already deployed on Sepolia testnet.

#### Deployed Contract

| Property | Value |
|:--|:--|
| **Contract** | `PRISMRegistry` |
| **Address** | `0xD96fF6ba63E803C877c82999C0C7DD829C3415B8` |
| **Chain** | Sepolia (Chain ID: 11155111) |
| **Standard** | ERC-5192 (Soulbound) |

#### Verify Deployment

```bash
# Set variables
RPC="https://sepolia.infura.io/v3/YOUR_PROJECT_ID"
CONTRACT="0xD96fF6ba63E803C877c82999C0C7DD829C3415B8"

# Check contract owner
cast call $CONTRACT "owner()(address)" --rpc-url $RPC

# Check issuer
cast call $CONTRACT "issuer()(address)" --rpc-url $RPC
```

#### Deploying Your Own Contract (Optional)

```bash
cd prism-protocol/contracts

# Build
forge build

# Deploy to Sepolia
forge create src/PRISMRegistry.sol:PRISMRegistry \
  --rpc-url $RPC \
  --private-key $PRIVATE_KEY \
  --constructor-args "YOUR_ISSUER_ADDRESS"
```

#### Contract Interface

```solidity
// Key functions
function mintAttestation(address user, bytes32 proofHash, uint64 expiresAt) external;
function revokeAttestation(address user) external;
function isHuman(address user) external view returns (bool);
function attestations(address user) external view returns (Attestation memory);
```

---

### 4. Frontend

The frontend handles webcam capture, Chroma Challenge UI, and user interaction.

#### Installation

```bash
cd prism-protocol/Chroma-main

# If using npm/Node.js frontend
npm install
npm run dev
```

#### Chroma Challenge Flow

1. User connects wallet
2. Frontend requests webcam access
3. Chroma Challenge flashes R -> G -> B -> W sequence
4. Frames streamed to Vision Backend via WebSocket
5. Backend returns liveness scores
6. If passed, Security Core mints Soulbound Token
7. Frontend displays success + transaction hash

---

## Configuration Reference

### Complete Environment Variable Reference

```bash
# =============================================================================
# PYTHON VISION BACKEND
# =============================================================================
PRISM_HOST=0.0.0.0
PRISM_PORT=8000
SECURITY_CORE_URL=http://localhost:8080
SECURITY_CORE_API_KEY=prism-python-secret

# =============================================================================
# JAVA SECURITY CORE
# =============================================================================
# Database
SPRING_DATASOURCE_URL=jdbc:mysql://localhost:3306/prism_db
SPRING_DATASOURCE_USERNAME=prism
SPRING_DATASOURCE_PASSWORD=your_secure_password

# Blockchain
WEB3_RPC_URL=https://sepolia.infura.io/v3/YOUR_PROJECT_ID
WEB3_CONTRACT_ADDRESS=0xD96fF6ba63E803C877c82999C0C7DD829C3415B8
WEB3_WALLET_PRIVATEKEY=0xYOUR_PRIVATE_KEY

# Attestation
PRISM_ATTESTATION_TTLSECONDS=604800

# API Security
PYTHON_API_KEY=prism-python-secret
```

---

## API Reference

### Vision Backend Endpoints

#### Health Check

```bash
GET http://localhost:8000/health

Response:
{
  "status": "healthy",
  "version": "1.0.0"
}
```

#### WebSocket Frame Stream

```
WS ws://localhost:8000/ws/verify

# Send frames as base64 JSON:
{
  "frame": "base64_encoded_image",
  "timestamp": 1234567890,
  "screenColor": "RED"
}

# Receive liveness scores:
{
  "eyeScore": 0.95,
  "skinScore": 0.92,
  "pulseScore": 0.88,
  "flashScore": 0.97,
  "isHuman": true
}
```

### Security Core Endpoints

#### Verify and Mint Attestation

```bash
POST http://localhost:8080/api/verify/human
Headers:
  Content-Type: application/json
  X-API-KEY: prism-python-secret

Body:
{
  "wallet": "0x7dC0B644bBd9bbaeEC62bE50CAE3472919A7a653",
  "sessionId": "unique-session-id",
  "eyeScore": 0.95,
  "skinScore": 0.92,
  "pulseScore": 0.88,
  "flashScore": 0.97
}

Response (Success):
{
  "status": "VERIFIED",
  "blockchainTx": "0xabc123...",
  "proofHash": "0xdef456...",
  "tokenId": "12345",
  "expiresAtEpochSec": 1704067200
}

Response (Failure):
{
  "status": "FAILED",
  "reason": "Liveness score below threshold"
}
```

#### Force Re-verification

Use `?force=true` to revoke existing attestation and mint new one:

```bash
POST http://localhost:8080/api/verify/human?force=true
```

#### Check Attestation Status

```bash
GET http://localhost:8080/api/verify/status/{wallet}
Headers:
  X-API-KEY: prism-python-secret

Response:
{
  "wallet": "0x7dC0B644bBd9bbaeEC62bE50CAE3472919A7a653",
  "isVerified": true,
  "expiresAt": "2025-01-15T00:00:00Z",
  "tokenId": "12345"
}
```

---

## On-Chain Verification

### Query Attestation Status

```bash
RPC="https://sepolia.infura.io/v3/YOUR_PROJECT_ID"
CONTRACT="0xD96fF6ba63E803C877c82999C0C7DD829C3415B8"
USER="0x7dC0B644bBd9bbaeEC62bE50CAE3472919A7a653"

# Check if user is verified
cast call $CONTRACT "isHuman(address)(bool)" $USER --rpc-url $RPC
# Returns: true or false

# Get full attestation struct
cast call $CONTRACT "attestations(address)(uint64,uint64,bytes32,uint16,address)" $USER --rpc-url $RPC
# Returns: (issuedAt, expiresAt, proofHash, score, issuer)
```

### Integration Example (Solidity)

```solidity
interface IPRISMRegistry {
    function isHuman(address user) external view returns (bool);
}

contract MyDAO {
    IPRISMRegistry public prism;
    
    constructor(address _prismRegistry) {
        prism = IPRISMRegistry(_prismRegistry);
    }
    
    modifier onlyHumans() {
        require(prism.isHuman(msg.sender), "PRISM: Not verified human");
        _;
    }
    
    function vote(uint256 proposalId) external onlyHumans {
        // Only verified humans can vote
    }
}
```

### Integration Example (JavaScript/ethers.js)

```javascript
import { ethers } from 'ethers';

const PRISM_REGISTRY = "0xD96fF6ba63E803C877c82999C0C7DD829C3415B8";
const ABI = ["function isHuman(address) view returns (bool)"];

const provider = new ethers.JsonRpcProvider(RPC_URL);
const prism = new ethers.Contract(PRISM_REGISTRY, ABI, provider);

async function checkHuman(walletAddress) {
    const isHuman = await prism.isHuman(walletAddress);
    console.log(`${walletAddress} is human: ${isHuman}`);
    return isHuman;
}
```

---

## Troubleshooting

### Common Issues

#### 1. `FAILED_TO_WRITE` in API Response

**Cause:** Blockchain transaction failed.

**Solutions:**
- Verify `WEB3_RPC_URL` is correct and accessible
- Verify `WEB3_WALLET_PRIVATEKEY` is correct (include `0x` prefix)
- Verify wallet has sufficient Sepolia ETH for gas
- Check RPC rate limits

```bash
# Test RPC connectivity
cast block-number --rpc-url $WEB3_RPC_URL

# Check wallet balance
cast balance $WALLET_ADDRESS --rpc-url $WEB3_RPC_URL
```

#### 2. `execution reverted` Error

**Cause:** Smart contract rejected the transaction.

**Common reasons:**
- Wallet already has active attestation (not expired)
- Caller is not authorized issuer

**Solution:** Use `?force=true` parameter to revoke and re-mint:

```bash
curl -X POST "http://localhost:8080/api/verify/human?force=true" ...
```

#### 3. `401 Unauthorized`

**Cause:** Missing or incorrect API key.

**Solution:** Include correct header:

```bash
-H "X-API-KEY: prism-python-secret"
```

#### 4. Database Connection Error

**Cause:** MariaDB not running or credentials incorrect.

**Solutions:**

```bash
# Check MariaDB status
sudo systemctl status mariadb

# Test connection
mysql -u prism -p -e "SELECT 1"

# Verify environment variables
echo $SPRING_DATASOURCE_URL
echo $SPRING_DATASOURCE_USERNAME
```

#### 5. Low Liveness Scores

**Cause:** Poor lighting, camera quality, or user not following instructions.

**Solutions:**
- Ensure adequate, even lighting on user's face
- User should face camera directly
- Webcam should be at least 720p
- User should remain still during rPPG capture (5 seconds)

---

## Security Considerations

### Private Key Management

- **Never** commit private keys to version control
- Use environment variables or secure secret management (e.g., HashiCorp Vault)
- For production, use hardware security modules (HSM) or cloud KMS
- Rotate keys periodically

### API Security

- The `X-API-KEY` header protects inter-service communication
- In production, use proper secret rotation
- Consider mTLS between services
- Rate limit public endpoints

### Smart Contract Security

- Contract ownership should be transferred to a multisig for production
- Consider timelocks for admin functions
- Monitor for unusual minting patterns
- Implement circuit breakers for emergency pause

### Data Privacy

- Video frames are processed in memory and discarded
- No biometric data is stored or transmitted
- Only cryptographic proofs are written on-chain
- Session data should be purged after verification

---

## Support

For issues, questions, or contributions:

- Review the [Whitepaper](WHITEPAPER.md) for technical details
- Check the [Technical Spec](PRISM_PROTOCOL.md) for implementation details
- Open an issue on the repository

---

<div align="center">

**PRISM Protocol**

*Physics Cannot Be Faked.*

</div>
