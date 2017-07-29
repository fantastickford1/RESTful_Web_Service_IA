package training;

import processing.core.PApplet;
import psvm.SVM;
import psvm.SVMProblem;

public class LibSVM extends Classifier{

    private SVM svmClassifier;

    public LibSVM(int numFeatures){
        super(numFeatures);
    }

    float[] doubleToFloat(double[] input) {

        float[] result = new float[input.length];

        for(int i = 0; i < input.length; i++){
            result[i] = (float) input[i];
        }

        return result;
    }

    @Override
    public void train() {

        float[][] trainingVectors = new float[trainingSamples.size()][trainingSamples.get(0).getFeatureVector().length];

        int[] labels = new int[trainingSamples.size()];

        for (int i = 0; i < trainingSamples.size(); i++) {
            trainingVectors[i] = doubleToFloat(trainingSamples.get(i).getFeatureVector());
            labels[i] = trainingSamples.get(i).getLabel();
        }

        svmClassifier = new SVM(new PApplet());

        svmClassifier.params.kernel_type = SVM.RBF_KERNEL;

        SVMProblem problem = new SVMProblem();

        problem.setNumFeatures(numFeatures);
        problem.setSampleData(labels, trainingVectors);

        svmClassifier.train(problem);
    }

    public void save(String filename){
        svmClassifier.saveModel(filename);
    }

    public void load(String filename){
        svmClassifier.loadModel(filename, numFeatures);
    }

    @Override
    double predict(Sample sample) {
        return svmClassifier.test(doubleToFloat(sample.getFeatureVector()));
    }

    public double predict(Sample sample, double[] confidence) {
        return svmClassifier.test(doubleToFloat(sample.getFeatureVector()), confidence);
    }

}
