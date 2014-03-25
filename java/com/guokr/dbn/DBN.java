package com.guokr.dbn;

import static com.guokr.dbn.MatrixUtils.opSigmoid;
import static com.guokr.dbn.MatrixUtils.opSoftmax;
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
            this.sig_layers[i] = new SigmoidLayer(isize, osize);
            // this.rbm_layers[i] = new RBlzmMLayer(isize, osize,
            // this.sig_layers[i].weights);
            this.rbm_layers[i] = new RBlzmMLayer(isize, osize, null);
        }

        this.log_layer = new LogRgrsLayer(lsizes[lsizes.length - 2], this.onum);
    }

    private AVector biased(AVector input) {
        AVector b = Vectorz.newVector(input.length() + 1);
        input.copyTo(b, 1);
        return b;
    }

    public void pretrain(int k, int epochs, double learning_rate, AVector input) {
        input = biased(input);

        AVector icur, iprev;

        for (int i = 0; i < lnum - 2; i++) { // layer-wise
            for (int epoch = 0; epoch < epochs; epoch++) { // training epochs

                icur = input;

                for (int l = 1; l <= i; l++) {
                    iprev = icur.clone();

                    icur = biased(Vectorz.newVector(lsizes[l]));
                    sig_layers[l - 1].osample_under_i(icur, iprev);
                }

                rbm_layers[i].contrastive_divergence(k, learning_rate, icur);
            }
        }
    }

    public void finetune(int epochs, double learning_rate, AVector input, AVector result) {
        input = biased(input);

        AVector icur, iprev;
        for (int epoch = 0; epoch < epochs; epoch++) {
            iprev = input;
            icur = biased(Vectorz.newVector(lsizes[1]));
            sig_layers[0].osample_under_i(icur, iprev);

            for (int i = 1; i < lnum - 2; i++) {
                iprev = icur.clone();
                icur = biased(Vectorz.newVector(lsizes[i + 1]));
                sig_layers[i].osample_under_i(icur, iprev);
            }

            log_layer.train(learning_rate, icur, result);
        }
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
