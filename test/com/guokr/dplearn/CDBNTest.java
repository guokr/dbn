package com.guokr.dplearn;

import static java.lang.Math.PI;
import static java.lang.Math.cos;
import static java.lang.Math.random;
import static java.lang.Math.sin;
import mikera.vectorz.AVector;
import mikera.vectorz.Vectorz;

import org.junit.Assert;
import org.junit.Test;

public class CDBNTest {

    public double s(double x) {
        return (7 + x) / 14;
    }

    public AVector key(double p, double q) {
        return Vectorz.create(Math.round(p), Math.round(q));
    }

    public AVector bitorus(double p, double q) {
        double theta = 2 * PI * random();
        double phi = 2 * PI * random();

        double R = 5;
        double r = 0.5;

        // torus
        // https://en.wikipedia.org/wiki/Torus
        double x = (R + (r + p) * cos(phi)) * cos(theta);
        double y = (R + (r + p) * cos(phi)) * sin(theta);
        double z = (r + p) * sin(phi);

        double u = (R + (r + q) * cos(theta)) * cos(phi);
        double v = (R + (r + q) * cos(theta)) * sin(phi);
        double w = (r + q) * sin(theta);

        return Vectorz.create(s(x), s(y), s(z), s(u), s(v), s(w));
    }

    @Test
    public void test() {

        int[] sizes_per_layer = { 6, 12, 24, 12, 6, 2 };
        CDBN cdbn = new CDBN(sizes_per_layer);

        // pretrain

        int k = 1;
        double pretrain_lr = 0.01;
        int pretraining_epochs = 10000;

        for (int i = 0; i < pretraining_epochs; i++) {
            cdbn.pretrain(k, pretrain_lr, bitorus(random(), random()));
        }

        // finetune

        double finetune_lr = 0.01;
        int finetune_epochs = 10000;

        for (int i = 0; i < finetune_epochs; i++) {
            double p = random();
            double q = random();
            cdbn.finetune(finetune_lr, bitorus(p, q), key(p, q));
        }

        // test

        for (int i = 0; i < 10; i++) {
            double p = random();
            double q = random();
            AVector input = bitorus(p, q);

            AVector output = cdbn.predict(input);
            AVector test = key(p, q);

            System.out.println("input:" + input);
            System.out.println("output:" + output);
            System.out.println("expected:" + test);
            System.out.println(test.epsilonEquals(output, 0.1));
            Assert.assertTrue("error is greater than expected!",
            test.epsilonEquals(output, 0.1));
        }
    }

}
