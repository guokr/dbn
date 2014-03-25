package com.guokr.dbn;

import static com.guokr.dbn.MatrixUtils.compose12;
import static com.guokr.dbn.MatrixUtils.one;
import static com.guokr.dbn.MatrixUtils.zero;
import static java.lang.Math.exp;
import mikera.matrixx.IMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.Vectorz;

public class LogRgrsLayer {
    public int     inum;
    public int     onum;
    public IMatrix weights;

    public LogRgrsLayer(int inum, int onum) {
        this.inum = inum;
        this.onum = onum;

        this.weights = compose12(one(onum, 1), zero(onum, inum));
    }

    private void softmax(AVector x) {
        double max = 0.0;
        double sum = 0.0;

        max = x.maxElement();
        for (int i = 0; i < onum; i++) {
            double cmp = exp(x.get(i) - max);
            x.set(i, cmp);
            sum += cmp;
        }

        x.scale(1 / sum);
    }

    public void train(double learning_rate, AVector x, AVector y) {
        AVector py_x = weights.transform(x);
        softmax(py_x);
        py_x.scale(-1);

        AVector dy = Vectorz.create(y);
        dy.add(py_x);
        dy.scale(learning_rate);

        this.weights.add(dy.outerProduct(x));
        for (int i = 0; i < onum; i++) {
            this.weights.set(i, 0, 1);
        }
    }

    public void predict(AVector x, AVector y) {
        y.set(weights.transform(x));
        softmax(y);
    }

}
