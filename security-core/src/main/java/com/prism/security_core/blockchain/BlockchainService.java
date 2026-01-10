package com.prism.security_core.blockchain;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.web3j.crypto.Credentials;
import org.web3j.protocol.Web3j;
import org.web3j.protocol.core.methods.response.TransactionReceipt;
import org.web3j.tx.RawTransactionManager;
import org.web3j.tx.gas.DefaultGasProvider;

@Service
public class BlockchainService {

    private final Web3j web3j;

    @Value("${web3.wallet.privateKey}")
    private String privateKey;

    @Value("${web3.contract.address}")
    private String contractAddress;

    public BlockchainService(Web3j web3j) {
        this.web3j = web3j;
    }

    public String storeProof(String data) throws Exception {

        // 1️⃣ Create SHA-256 hash (32 bytes)
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] hashBytes = md.digest(data.getBytes(StandardCharsets.UTF_8));

        // 2️⃣ ENSURE exactly 32 bytes
        byte[] bytes32 = Arrays.copyOf(hashBytes, 32);

        // 3️⃣ Wallet
        Credentials creds = Credentials.create(privateKey);

        // 4️⃣ Tx manager
        RawTransactionManager txManager = new RawTransactionManager(web3j, creds);

        // 5️⃣ Contract
        PrismVerificationContract contract = PrismVerificationContract.load(
                contractAddress,
                web3j,
                txManager,
                new DefaultGasProvider());

        // 6️⃣ Send tx
        TransactionReceipt receipt = contract.storeVerification(bytes32).send();

        return receipt.getTransactionHash();
    }

    // private String sha256(String input) throws Exception {
    //     MessageDigest md = MessageDigest.getInstance("SHA-256");
    //     byte[] hash = md.digest(input.getBytes(StandardCharsets.UTF_8));
    //     StringBuilder sb = new StringBuilder();
    //     for (byte b : hash)
    //         sb.append(String.format("%02x", b));
    //     return sb.toString();
    // }
}
