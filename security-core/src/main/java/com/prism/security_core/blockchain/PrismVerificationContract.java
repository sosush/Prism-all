package com.prism.security_core.blockchain;

import java.util.Arrays;
import java.util.Collections;

import org.web3j.abi.datatypes.Function;
import org.web3j.abi.datatypes.generated.Bytes32;
import org.web3j.protocol.Web3j;
import org.web3j.tx.Contract;
import org.web3j.tx.TransactionManager;
import org.web3j.tx.gas.ContractGasProvider;

public class PrismVerificationContract extends Contract {

    protected PrismVerificationContract(String contractAddress, Web3j web3j,
            TransactionManager transactionManager,
            ContractGasProvider gasProvider) {

        super("", contractAddress, web3j, transactionManager, gasProvider);
    }

    public static PrismVerificationContract load(
            String contractAddress,
            Web3j web3j,
            TransactionManager transactionManager,
            ContractGasProvider gasProvider) {

        return new PrismVerificationContract(
                contractAddress, web3j, transactionManager, gasProvider);
    }

    public org.web3j.protocol.core.RemoteFunctionCall<org.web3j.protocol.core.methods.response.TransactionReceipt> storeVerification(
            byte[] hash) {

        Function function = new Function(
                "storeVerification",
                Arrays.asList(new Bytes32(hash)),
                Collections.emptyList());

        return executeRemoteCallTransaction(function);
    }
}
