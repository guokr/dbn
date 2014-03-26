package com.guokr.dbn;

import static com.guokr.dbn.MatrixUtils.compose12;
import static com.guokr.dbn.MatrixUtils.opSoftmax;
import static com.guokr.dbn.MatrixUtils.zero;
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

        this.weights = compose12(zero(onum, 1), zero(onum, inum));
    }

    public void train(double learning_rate, AVector x, AVector y) {
        x.set(0, 1);

        AVector py_x = weights.transform(x);
        py_x.applyOp(opSoftmax(py_x));
        py_x.scale(-1);

        AVector dy = Vectorz.create(y);
        dy.add(py_x);
        dy.scale(learning_rate);

        this.weights.add(dy.outerProduct(x));
    }

    public void predict(AVector x, AVector y) {
        y.set(weights.transform(x));
        y.applyOp(opSoftmax(y));
    }

}
