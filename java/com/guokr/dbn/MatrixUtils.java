package com.guokr.dbn;

import static com.guokr.dbn.MathUtils.uniform;
import mikera.matrixx.AMatrix;
import mikera.matrixx.IMatrix;
import mikera.matrixx.Matrixx;

public class MatrixUtils {

    public static IMatrix zero(int rows, int columns) {
        AMatrix m = Matrixx.newMatrix(rows, columns);
        m.fill(0);
        return m;
    }

    public static IMatrix constant(int rows, int columns, double x) {
        AMatrix m = Matrixx.newMatrix(rows, columns);
        m.fill(x);
        return m;
    }

    public static IMatrix random(int rows, int columns, double min, double max) {
        AMatrix m = Matrixx.newMatrix(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                m.set(i, j, uniform(min, max));
            }
        }
        return m;
    }

    public static IMatrix compose21(IMatrix m11, IMatrix m21) {
        int r1 = m11.rowCount(), r2 = m21.rowCount(), r = r1 + r2;
        int c = m11.columnCount();
        AMatrix m = Matrixx.newMatrix(r, c);
        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c; j++) {
                m.set(i, j, m11.get(i, j));
            }
        }
        for (int i = 0; i < r2; i++) {
            for (int j = 0; j < c; j++) {
                m.set(r1 + i, j, m21.get(i, j));
            }
        }
        return m;
    }

    public static IMatrix compose22(IMatrix m11, IMatrix m12, IMatrix m21, IMatrix m22) {
        int r1 = m11.rowCount(), r2 = m22.rowCount(), r = r1 + r2;
        int c1 = m11.columnCount(), c2 = m22.columnCount(), c = c1 + c2;
        AMatrix m = Matrixx.newMatrix(r, c);
        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c1; j++) {
                m.set(i, j, m11.get(i, j));
            }
        }
        for (int i = 0; i < r2; i++) {
            for (int j = 0; j < c1; j++) {
                m.set(r1 + i, j, m21.get(i, j));
            }
        }
        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c2; j++) {
                m.set(i, c1 + j, m12.get(i, j));
            }
        }
        for (int i = 0; i < r2; i++) {
            for (int j = 0; j < c2; j++) {
                m.set(r1 + i, c1 + j, m22.get(i, j));
            }
        }
        return m;
    }

}
