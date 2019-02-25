package chapter_5;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;
import java.util.List;
import java.util.Random;

public class NetworkTrainedToSumNumbersUsingRegression {
  private static final int seed = 1244;
  private static final int nEpochs = 250;
  private static final int nSamples = 2000;
  private static final int batchSize = 80;
  private static final double learningRate = 0.01;
  private static int MIN_RANGE = 0;
  private static int MAX_RANGE = 3;

  private static final Random rng = new Random(seed);

  public static void main(String[] args) {
    DataSetIterator iterator = generateTrainingSumData(batchSize, rng);

    MultiLayerNetwork net = configureMultiLayer();

    trainModel(iterator, net);

    testAbilityOfModelToSumNumbers(net);

  }

  private static void testAbilityOfModelToSumNumbers(MultiLayerNetwork net) {
    final INDArray input = Nd4j.create(new double[]{0.111111, 0.3333333333333}, new int[]{1, 2});
    INDArray out = net.output(input, false);
    System.out.println(out);

    final INDArray input2 = Nd4j.create(new double[]{0.1, 0.8}, new int[]{1, 2});
    INDArray out2 = net.output(input2, false);
    System.out.println(out2);
  }

  private static void trainModel(DataSetIterator iterator, MultiLayerNetwork net) {
    for (int i = 0; i < nEpochs; i++) {
      iterator.reset();
      net.fit(iterator);
    }
  }

  @NotNull
  private static MultiLayerNetwork configureMultiLayer() {
    //Create the network
    int numInput = 2;
    int numOutputs = 1;
    int nHidden = 10;
    MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
        .seed(seed)
        .weightInit(WeightInit.XAVIER)
        .updater(new Nesterovs(learningRate, 0.9))
        .list()
        .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
            .activation(Activation.TANH)
            .build())
        .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
            .activation(Activation.IDENTITY)
            .nIn(nHidden).nOut(numOutputs).build())
        .build()
    );
    net.init();
    net.setListeners(new ScoreIterationListener(1));
    return net;
  }

  private static DataSetIterator generateTrainingSumData(int batchSize, Random rand) {
    double[] sum = new double[nSamples];
    double[] input1 = new double[nSamples];
    double[] input2 = new double[nSamples];
    for (int i = 0; i < nSamples; i++) {
      input1[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble();
      input2[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble();
      sum[i] = input1[i] + input2[i];
    }
    INDArray inputNDArray1 = Nd4j.create(input1, new int[]{nSamples, 1});
    INDArray inputNDArray2 = Nd4j.create(input2, new int[]{nSamples, 1});
    INDArray inputNDArray = Nd4j.hstack(inputNDArray1, inputNDArray2);
    INDArray outPut = Nd4j.create(sum, new int[]{nSamples, 1});
    DataSet dataSet = new DataSet(inputNDArray, outPut);
    List<DataSet> listDs = dataSet.asList();
    Collections.shuffle(listDs, rng);
    System.out.println("data used to train network:" + listDs);
    return new ListDataSetIterator<>(listDs, batchSize);

  }
}
