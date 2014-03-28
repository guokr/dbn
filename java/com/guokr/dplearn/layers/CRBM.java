package com.guokr.dplearn.layers;

import mikera.matrixx.AMatrix;
import mikera.vectorz.AVector;

import com.guokr.dplearn.util.MathUtils;

public class CRBM extends RBM {

    public CRBM(int vnum, int hnum, AMatrix weights) {
        super(vnum, hnum, weights);
    }

    public double down(AVector hsample, AVector hweight) {
        return hsample.innerProduct(hweight).value;
    }

    private double a(double x) {
        return 1 / (1 - Math.pow(Math.E, -x)) - 1 / x;
    }

    private double b(double x) {
        return Math.log(1 - (1 - Math.pow(Math.E, x)) * MathUtils.uniform(0, 1)) / (x + 1e-14);
    }

    public void vsample_under_h(AVector vsample, AVector vmean, AVector hsample) {
        for (int i = 0; i < vnum + 1; i++) {
            double h = down(hsample, weights.getRow(i));
            vmean.set(i, a(h));
            vsample.set(i, b(h));
        }
        vmean.set(0, 1);
        vsample.set(0, 1);
    }

}
