package com.guokr.dbn;

import static com.guokr.dbn.ANNUtils.biased;
import static com.guokr.dbn.MatrixUtils.opSigmoid;
import static com.guokr.dbn.MatrixUtils.opSoftmax;
import static com.guokr.dbn.MatrixUtils.transpose;
import mikera.vectorz.AVector;
import mikera.vectorz.Vectorz;

public class DBN {

    public int            inum;
    public int            onum;

    public int            lnum;
    public int[]          lsizes;

    public SigmoidLayer[] sig_layers;
    public RBlzmMLayer[]  rbm_layers;
    public LogRgrsLayer   log_layer;

    public DBN(int[] lsizes) {
        this.inum = lsizes[0];
        this.onum = lsizes[lsizes.length - 1];
        this.lnum = lsizes.length;
        this.lsizes = lsizes;

        this.sig_layers = new SigmoidLayer[this.lnum - 2];
        this.rbm_layers = new RBlzmMLayer[this.lnum - 2];

        for (int i = 0; i < this.lnum - 2; i++) {
            int isize = lsizes[i];
            int osize = lsizes[i + 1];

            SigmoidLayer sigmoidLayer = new SigmoidLayer(isize, osize);
            this.sig_layers[i] = sigmoidLayer;
            this.rbm_layers[i] = new RBlzmMLayer(isize, osize, transpose(sigmoidLayer.weights));
        }

        this.log_layer = new LogRgrsLayer(lsizes[lsizes.length - 2], this.onum);
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
            SigmoidLayer lcur = sig_layers[i];

            icur = lcur.weights.transform(iprev);
            icur.applyOp(opSigmoid);

            iprev = icur;
        }

        AVector result = log_layer.weights.transform(icur);
        result.applyOp(opSoftmax(result));

        return result;
    }
}
