package com.guokr.dplearn.layers;

import static com.guokr.dplearn.util.ANNUtils.biased;
import static com.guokr.dplearn.util.MathUtils.binomial;
import static com.guokr.dplearn.util.MathUtils.sigmoid;
import static com.guokr.dplearn.util.MatrixUtils.compose22;
import static com.guokr.dplearn.util.MatrixUtils.random;
import static com.guokr.dplearn.util.MatrixUtils.tensorProduct;
import static com.guokr.dplearn.util.MatrixUtils.zero;
import mikera.matrixx.AMatrix;
import mikera.matrixx.IMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.Vectorz;

import com.guokr.dplearn.util.MathUtils;

public class CRBM {

    public int     vnum;
    public int     hnum;
    public AMatrix weights;

    public CRBM(int vnum, int hnum, AMatrix weights) {
        this.vnum = vnum;
        this.hnum = hnum;

        if (weights != null) {
            this.weights = weights;
        } else {
            double alpha = 1.0 / this.vnum;
            IMatrix rand = random(vnum, hnum, -alpha, alpha);

            this.weights = compose22(zero(1, 1), zero(1, hnum), zero(vnum, 1), rand);
        }
    }

    public double up(AVector vsample, AVector vweight) {
        return vsample.innerProduct(vweight).value;
    }

    public double down(AVector hsample, AVector hweight) {
        return hsample.innerProduct(hweight).value;
    }

    public void hsample_under_v(AVector hsample, AVector hmean, AVector vsample) {
        for (int i = 0; i < hnum + 1; i++) {
            double v = up(vsample, weights.getColumn(i));
            hmean.set(i, sigmoid(v));
            hsample.set(i, binomial(1, sigmoid(v)));
        }
        hmean.set(0, 1);
        hsample.set(0, 1);
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

    public void gibbs_hvh(AVector phsample, AVector nvmeans, AVector nvsamples, AVector nhmeans, AVector nhsamples) {
        vsample_under_h(nvsamples, nvmeans, phsample);
        hsample_under_v(nhsamples, nhmeans, nvsamples);
    }

    public void contrastive_divergence(int k, double learning_rate, AVector input) {
        AVector phmean = biased(hnum);
        AVector phsample = biased(hnum);

        AVector nvmeans = biased(vnum);
        AVector nvsamples = biased(vnum);

        AVector nhmeans = biased(hnum);
        AVector nhsamples = biased(hnum);

        hsample_under_v(phsample, phmean, input);

        for (int step = 0; step < k; step++) {
            if (step == 0) {
                gibbs_hvh(phsample, nvmeans, nvsamples, nhmeans, nhsamples);
            } else {
                gibbs_hvh(nhsamples, nvmeans, nvsamples, nhmeans, nhsamples);
            }
        }

        IMatrix mp = tensorProduct(input, phmean);
        IMatrix mn = tensorProduct(nvsamples, nhmeans);

        mp.scale(learning_rate);
        mn.scale(-learning_rate);

        weights.add(mp);
        weights.add(mn);
    }

    public void reconstruct(AVector vrecons, AVector vsample) {
        AVector h = Vectorz.newVector(hnum + 1);

        for (int i = 0; i < hnum + 1; i++) {
            h.set(i, up(vsample, weights.getColumn(i)));
        }

        for (int i = 0; i < vnum + 1; i++) {
            vrecons.set(i, sigmoid(weights.getRow(i).innerProduct(h).value));
        }
    }

}
