package com.guokr.dbn;

import static com.guokr.dbn.MatrixUtils.zero;
import mikera.vectorz.AVector;
import mikera.vectorz.Vectorz;

public class ANNUtils {

    public static AVector biased(AVector input) {
        AVector b = Vectorz.newVector(input.length() + 1);
        b.set(0, 1);
        input.copyTo(b, 1);
        return b;
    }

    public static AVector biased(int dim) {
        return biased(zero(dim));
    }

}
