package com.prism.security_core.blockchain;

import java.math.BigInteger;

import org.web3j.tx.gas.ContractGasProvider;

public class PolygonGasProvider implements ContractGasProvider {

    private static final BigInteger GAS_PRICE = BigInteger.valueOf(35_000_000_000L); // 35 Gwei

    private static final BigInteger GAS_LIMIT = BigInteger.valueOf(300_000);

    @Override
    public BigInteger getGasPrice(String contractFunc) {
        return GAS_PRICE;
    }

    @Override
    public BigInteger getGasPrice() {
        return GAS_PRICE;
    }

    @Override
    public BigInteger getGasLimit(String contractFunc) {
        return GAS_LIMIT;
    }

    @Override
    public BigInteger getGasLimit() {
        return GAS_LIMIT;
    }
}
