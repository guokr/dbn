package com.guokr.dplearn;

import static com.guokr.dplearn.util.ANNUtils.biased;
import static com.guokr.dplearn.util.MatrixUtils.opSigmoid;
import static com.guokr.dplearn.util.MatrixUtils.opSoftmax;
import static com.guokr.dplearn.util.MatrixUtils.transpose;
import mikera.vectorz.AVector;
import mikera.vectorz.Vectorz;

import com.guokr.dplearn.layers.CRBM;
import com.guokr.dplearn.layers.LogR;
import com.guokr.dplearn.layers.Sigmd;

public class CDBN {

    public int     inum;
    public int     onum;

    public int     lnum;
    public int[]   lsizes;

    public Sigmd[] sig_layers;
    public CRBM[]  rbm_layers;
    public LogR    log_layer;

    public CDBN(int[] lsizes) {
        this.inum = lsizes[0];
        this.onum = lsizes[lsizes.length - 1];
        this.lnum = lsizes.length;
        this.lsizes = lsizes;

        this.sig_layers = new Sigmd[this.lnum - 2];
        this.rbm_layers = new CRBM[this.lnum - 2];

        for (int i = 0; i < this.lnum - 2; i++) {
            int isize = lsizes[i];
            int osize = lsizes[i + 1];

            Sigmd sigmoidLayer = new Sigmd(isize, osize);
            this.sig_layers[i] = sigmoidLayer;
            this.rbm_layers[i] = new CRBM(isize, osize, transpose(sigmoidLayer.weights));
        }

        this.log_layer = new LogR(lsizes[lsizes.length - 2], this.onum);
    }

    public void pretrain(int k, double learning_rate, AVector input) {
        AVector icur = null, iprev = null;
        for (int i = 0; i < lnum - 2; i++) { // layer-wise
            for (int l = 0; l <= i; l++) {
                if (l == 0) {
                    icur = biased(input);
                } else {
                    iprev = icur.clone();

                    icur = biased(lsizes[l]);
                    sig_layers[l - 1].osample_under_i(icur, iprev);
                }
            }

            rbm_layers[i].contrastive_divergence(k, learning_rate, icur);
        }
    }

    public void finetune(double learning_rate, AVector input, AVector result) {
        input = biased(input);

        AVector icur, iprev;
        iprev = input;
        icur = biased(lsizes[1]);
        sig_layers[0].osample_under_i(icur, iprev);

        for (int i = 1; i < lnum - 2; i++) {
            iprev = icur.clone();
            icur = biased(Vectorz.newVector(lsizes[i + 1]));
            sig_layers[i].osample_under_i(icur, iprev);
        }

        log_layer.train(learning_rate, icur, result);
    }

    public AVector predict(AVector input) {
        input = biased(input);

        AVector icur = null, iprev = input;

        for (int i = 0; i < lnum - 2; i++) {
            Sigmd lcur = sig_layers[i];

            icur = lcur.weights.transform(iprev);
            icur.applyOp(opSigmoid);

            iprev = icur;
        }

        AVector result = log_layer.weights.transform(icur);
        result.applyOp(opSoftmax(result));

        return result;
    }
}
