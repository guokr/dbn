package com.guokr.dplearn;

import static java.lang.Math.*;

import mikera.vectorz.AVector;
import mikera.vectorz.Vectorz;

import org.junit.Assert;
import org.junit.Test;

import com.guokr.dplearn.DBN;

public class CDBNTest {

    public double s(double x) {
        return (1 + x) / 2;
    }

    public AVector knots(double p, double q) {
        double theta = 2 * PI * p;
        double phi = 2 * PI * q;

        // (2,3)-torus knot or Trefoil knot
        // https://en.wikipedia.org/wiki/Torus_knot
        // https://en.wikipedia.org/wiki/Trefoil_knot
        double x = (sin(theta) + 2 * sin(2 * theta)) / 3;
        double y = (cos(theta) - 2 * cos(2 * theta)) / 3;
        double z = -sin(3 * theta);

        // (2,3)-torus knot
        // https://en.wikipedia.org/wiki/Torus_knot
        double u = ((cos(3 * phi) + 2) * cos(2 * phi)) / 3;
        double v = ((cos(3 * phi) + 2) * sin(2 * phi)) / 3;
        double w = -sin(3 * phi);

        return Vectorz.create(s(x), s(y), s(z), s(u), s(v), s(w));
    }

    public AVector kleinbottle(double p, double q) {
        double theta = 2 * PI * p;
        double phi = 2 * PI * q;

        double x = (cos(theta / 2) * cos(phi) - sin(theta / 2) * sin(2 * phi)) / 2;
        double y = (sin(theta / 2) * cos(phi) - cos(theta / 2) * sin(2 * phi)) / 2;
        double z = cos(theta) * (1 + sin(phi)) / 2;
        double w = sin(theta) * (1 + sin(phi)) / 2;

        return Vectorz.create(s(x), s(y), s(z), s(w));
    }

    @Test
    public void test() {

        int[] sizes_per_layer = { 4, 8, 16, 8, 4, 2 };
        DBN dbn = new DBN(sizes_per_layer);

        // pretrain

        int k = 1;
        double pretrain_lr = 0.01;
        int pretraining_epochs = 10000;

        for (int i = 0; i < pretraining_epochs; i++) {
            dbn.pretrain(k, pretrain_lr, kleinbottle(random(), random()));
        }

        // finetune

        double finetune_lr = 0.01;
        int finetune_epochs = 1000;

        for (int i = 0; i < finetune_epochs; i++) {
            double p = random();
            double q = random();
            AVector key = kleinbottle(p, q);
            AVector val = Vectorz.create(p, q);
            dbn.finetune(finetune_lr, key, val);
        }

        // test

        for (int i = 0; i < 100; i++) {
            double p = random();
            double q = random();
            AVector input = kleinbottle(p, q);

            AVector output = dbn.predict(input);
            AVector test = Vectorz.create(p, q);

            System.out.println("result:" + output + ", expected:" + test);
            Assert.assertTrue("error is greater than expected!", test.epsilonEquals(output, 0.1));
        }
    }

}
