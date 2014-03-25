package com.guokr.dbn;

import static com.guokr.dbn.MathUtils.binomial;
import static com.guokr.dbn.MathUtils.sigmoid;
import static com.guokr.dbn.MatrixUtils.compose22;
import static com.guokr.dbn.MatrixUtils.random;
import static com.guokr.dbn.MatrixUtils.zero;
import mikera.matrixx.IMatrix;
import mikera.vectorz.AVector;

public class SigmoidLayer {
    public int     inum;
    public int     onum;
    public IMatrix weights;

    public SigmoidLayer(int inum, int onum) {
        this.inum = inum;
        this.onum = onum;

        double alpha = 1.0 / this.inum;
        IMatrix rand = random(onum, inum, -alpha, alpha);

        this.weights = compose22(zero(1, 1), zero(1, inum), zero(onum, 1), rand);
    }

    public void osample_under_i(AVector osample, AVector isample) {
        for (int i = 0; i < onum; i++) {
            osample.set(i, binomial(1, sigmoid(isample.innerProduct(weights.getRow(i)).value)));
        }
    }
}
