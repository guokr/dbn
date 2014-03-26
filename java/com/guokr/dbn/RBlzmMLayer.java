package com.guokr.dbn;

import static com.guokr.dbn.ANNUtils.biased;
import static com.guokr.dbn.MathUtils.binomial;
import static com.guokr.dbn.MathUtils.sigmoid;
import static com.guokr.dbn.MatrixUtils.compose22;
import static com.guokr.dbn.MatrixUtils.random;
import static com.guokr.dbn.MatrixUtils.tensorProduct;
import static com.guokr.dbn.MatrixUtils.zero;
import mikera.matrixx.AMatrix;
import mikera.matrixx.IMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.Vectorz;

public class RBlzmMLayer {

    public int     vnum;
    public int     hnum;
    public AMatrix weights;

    public RBlzmMLayer(int vnum, int hnum, AMatrix weights) {
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
        return sigmoid(vsample.innerProduct(vweight).value);
    }

    public double down(AVector hsample, AVector hweight) {
        return sigmoid(hsample.innerProduct(hweight).value);
    }

    public void hsample_under_v(AVector hsample, AVector hmean, AVector vsample) {
        for (int i = 0; i < hnum + 1; i++) {
            hmean.set(i, up(vsample, weights.getColumn(i)));
            hsample.set(i, binomial(1, hmean.get(i)));
        }
        hmean.set(0, 1);
        hsample.set(0, 1);
    }

    public void vsample_under_h(AVector vsample, AVector vmean, AVector hsample) {
        for (int i = 0; i < vnum + 1; i++) {
            vmean.set(i, down(hsample, weights.getRow(i)));
            vsample.set(i, binomial(1, vmean.get(i)));
        }
        vmean.set(0, 1);
        vsample.set(0, 1);
    }

    public void gibbs_hvh(AVector hpsample, AVector nvmeans, AVector nvsamples, AVector nhmeans, AVector nhsamples) {
        vsample_under_h(nvsamples, nvmeans, hpsample);
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

        System.out.println(phmean);
        System.out.println(nhmeans);

        mp.scale(learning_rate);
        mn.scale(-learning_rate);

        weights.add(mp);
        weights.add(mn);

        System.out.println(weights);
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
