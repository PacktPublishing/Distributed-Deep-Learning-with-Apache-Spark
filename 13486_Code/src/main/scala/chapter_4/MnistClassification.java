package chapter_4;

import chapter2.DataUtilities;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class MnistClassification {

  private static final Logger log = LoggerFactory.getLogger(MnistClassification.class);
  private static final String basePath = System.getProperty("java.io.tmpdir") + "/mnist";
  private static final String dataUrl = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

  public static void main(String[] args) throws Exception {
    int height = 28;
    int width = 28;
    int channels = 1;
    int outputNum = 10;
    int batchSize = 54;
    int iterations = 1;

    int seed = 1234;
    Random randNumGen = new Random(seed);

    downloadMnistIfNeeded();

    // vectorization of train data
    FileSplit trainSplit = loadTrainDataSet(randNumGen);
    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
    trainRR.initialize(trainSplit);
    DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);


    DataNormalization scaler = trainWithPixelValues(trainIter);

    // vectorization of test data
    DataSetIterator testIter = testDataSet(height, width, channels, outputNum, batchSize, randNumGen, labelMaker, scaler);

    MultiLayerNetwork net = configureNetwork(channels, outputNum, seed);
    log.debug("Total num of params: {}", net.numParams());

    performCrossValidation(iterations, trainIter, testIter, net);

  }

  private static void performCrossValidation(int iterations, DataSetIterator trainIter, DataSetIterator testIter, MultiLayerNetwork net) {
    for (int i = 0; i < iterations; i++) {
      net.fit(trainIter);
      log.info("Completed epoch {}", i);
      Evaluation eval = net.evaluate(testIter);
      log.info(eval.stats());
      trainIter.reset();
      testIter.reset();
    }
  }

  @NotNull
  private static MultiLayerNetwork configureNetwork(int channels, int outputNum, int seed) {
    log.info("Network configuration and training...");
    Map<Integer, Double> lrSchedule = new HashMap<>();
    lrSchedule.put(0, 0.06);
    lrSchedule.put(200, 0.05);
    lrSchedule.put(600, 0.028);
    lrSchedule.put(800, 0.0060);
    lrSchedule.put(1000, 0.001);

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .l2(0.0005)
        .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, lrSchedule)))
        .weightInit(WeightInit.XAVIER)
        .list()
        .layer(0, new ConvolutionLayer.Builder(5, 5)
            .nIn(channels)
            .stride(1, 1)
            .nOut(20)
            .activation(Activation.IDENTITY)
            .build())
        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2, 2)
            .stride(2, 2)
            .build())
        .layer(2, new ConvolutionLayer.Builder(5, 5)
            .stride(1, 1) // nIn need not specified in later layers
            .nOut(50)
            .activation(Activation.IDENTITY)
            .build())
        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2, 2)
            .stride(2, 2)
            .build())
        .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
            .nOut(500).build())
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(outputNum)
            .activation(Activation.SOFTMAX)
            .build())
        .setInputType(InputType.convolutionalFlat(28, 28, 1)) // InputType.convolutional for normal image
        .backprop(true).pretrain(false).build();

    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    net.setListeners(new ScoreIterationListener(10));
    return net;
  }

  @NotNull
  private static DataSetIterator testDataSet(int height, int width, int channels, int outputNum, int batchSize, Random randNumGen, ParentPathLabelGenerator labelMaker, DataNormalization scaler) throws IOException {
    File testData = new File(basePath + "/mnist_png/testing");
    FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
    ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
    testRR.initialize(testSplit);
    DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
    testIter.setPreProcessor(scaler); // same normalization for better results
    return testIter;
  }

  @NotNull
  private static DataNormalization trainWithPixelValues(DataSetIterator trainIter) {
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    scaler.fit(trainIter);
    trainIter.setPreProcessor(scaler);
    return scaler;
  }

  @NotNull
  private static FileSplit loadTrainDataSet(Random randNumGen) {
    File trainData = new File(basePath + "/mnist_png/training");
    return new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
  }

  private static void downloadMnistIfNeeded() throws IOException {
    log.info("Data load and vectorization...");
    String localFilePath = basePath + "/mnist_png.tar.gz";
    if (DataUtilities.downloadFile(dataUrl, localFilePath))
      log.debug("Data downloaded from {}", dataUrl);
    if (!new File(basePath + "/mnist_png").exists())
      DataUtilities.extractTarGz(localFilePath, basePath);
  }

}
