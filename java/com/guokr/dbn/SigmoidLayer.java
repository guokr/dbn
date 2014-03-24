package com.guokr.dbn;

import static com.guokr.dbn.MathUtils.binomial;
import static com.guokr.dbn.MathUtils.sigmoid;
import static com.guokr.dbn.MatrixUtils.compose21;
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
        IMatrix rand = random(inum, onum, -alpha, alpha);

        this.weights = compose21(zero(1, onum), rand);
    }

    public void osample_under_i(AVector osample, AVector isample) {
        for (int i = 0; i < onum; i++) {
            osample.set(i, binomial(1, sigmoid(isample.innerProduct(weights.getRow(i)).value)));
        }
    }
}
