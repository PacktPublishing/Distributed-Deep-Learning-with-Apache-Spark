package chapter2;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.net.URL;

/**
 * Given a movie review (raw text), classify that movie review as either positive or negative based on the words it contains.
 */
public class PredictCommentsUsingRNNAndWord2Vec {


    private static final int BATCH_SIZE = 64;
    private static final int SIZE_OF_VECTOR_IN_GOOGLE_NEWS_MODEL = 300;
    private static final int N_EPOCHS = 1;
    private static final int MAX_NUMBER_OF_WORDS_TAKEN_FROM_REVIEW = 256;
    private static final int SEED = 0;


    private static final String IMDB_COMMENTS_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
    private static final String IMDB_DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/");
    private static final String GOOGLE_NEWS_VECTOR_PATH = "/Users/tomaszlelek/Downloads/GoogleNews-vectors-negative300.bin.gz"; //# paste your PATH here
    //download from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit


    public static void main(String[] args) throws Exception {
        downloadIMDBDatabase();

        Nd4j.getMemoryManager().setAutoGcWindow(10000);

        //two output classes - first for positive comments about movie, second fir negative comments
        MultiLayerNetwork net = configureMultiLayerWithTwoOutputClasses();

        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(GOOGLE_NEWS_VECTOR_PATH));
        Word2VecTransformingIterator train = new Word2VecTransformingIterator(IMDB_DATA_PATH, wordVectors, BATCH_SIZE, MAX_NUMBER_OF_WORDS_TAKEN_FROM_REVIEW, true);
        Word2VecTransformingIterator test = new Word2VecTransformingIterator(IMDB_DATA_PATH, wordVectors, BATCH_SIZE, MAX_NUMBER_OF_WORDS_TAKEN_FROM_REVIEW, false);

        performTraining(net, train, test);
        printFirstPredictedPositiveReview(net, test);
    }

    private static void printFirstPredictedPositiveReview(MultiLayerNetwork net, Word2VecTransformingIterator test) throws IOException {
        //After training: load a single example and generate predictions
        File firstPositiveReviewFile = new File(FilenameUtils.concat(IMDB_DATA_PATH, "aclImdb/test/pos/0_10.txt"));
        String firstPositiveReview = FileUtils.readFileToString(firstPositiveReviewFile);

        INDArray features = test.transformStringIntoFeatureVectorOfNumber(firstPositiveReview, MAX_NUMBER_OF_WORDS_TAKEN_FROM_REVIEW);
        INDArray networkOutput = net.output(features);
        long timeSeriesLength = networkOutput.size(2);
        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

        System.out.println("\n\n-------------------------------");
        System.out.println("First positive review: \n" + firstPositiveReview);
        System.out.println("\n\nProbabilities at last time step:");
        System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
        System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));
    }

    private static void performTraining(MultiLayerNetwork net, Word2VecTransformingIterator train, Word2VecTransformingIterator test) {
        System.out.println("Starting training");
        for (int i = 0; i < N_EPOCHS; i++) {
            net.fit(train);
            train.reset();
            System.out.println("Epoch " + i + " complete. Starting evaluation:");

            //Run evaluation. This is on 25k reviews, so can take some time
            Evaluation evaluation = net.evaluate(test);
            System.out.println(evaluation.stats());
        }
    }

    @NotNull
    private static MultiLayerNetwork configureMultiLayerWithTwoOutputClasses() {
        //Set up network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(SEED)
            .updater(new Adam(5e-3))
            .l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .list()
            .layer(0, new LSTM.Builder().nIn(SIZE_OF_VECTOR_IN_GOOGLE_NEWS_MODEL).nOut(256)
                .activation(Activation.TANH).build())
            .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        return net;
    }

    private static void downloadIMDBDatabase() throws Exception {
        File directory = new File(IMDB_DATA_PATH);
        if(!directory.exists()) directory.mkdir();

        String archivePath = IMDB_DATA_PATH + "aclImdb_v1.tar.gz";
        File archiveFile = new File(archivePath);
        String extractedPath = IMDB_DATA_PATH + "aclImdb";
        File extractedFile = new File(extractedPath);

        if( !archiveFile.exists() ){
            System.out.println("Starting data download (80MB)...");
            FileUtils.copyURLToFile(new URL(IMDB_COMMENTS_URL), archiveFile);
            System.out.println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath());
            DataUtilities.extractTarGz(archivePath, IMDB_DATA_PATH);
        } else {
            System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
            if( !extractedFile.exists()){
                DataUtilities.extractTarGz(archivePath, IMDB_DATA_PATH);
            } else {
            	System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
            }
        }
    }


}
