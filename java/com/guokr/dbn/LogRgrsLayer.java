package com.guokr.dbn;

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

        this.weights = zero(inum + 1, onum);
    }

    public void train(double learning_rate, AVector x, AVector y) {
        AVector py_x = weights.transform(x);
        softmax(py_x);
        py_x.scale(-1);

        AVector dy = Vectorz.create(y);
        dy.add(py_x);
        dy.scale(learning_rate);

        this.weights.add(dy.outerProduct(x));
    }

    public void softmax(AVector x) {
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

    public void predict(AVector x, AVector y) {
        y.add(weights.transform(x));
        softmax(y);
    }

}
