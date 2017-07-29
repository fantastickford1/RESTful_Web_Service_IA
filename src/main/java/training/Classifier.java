package training;

import java.util.ArrayList;
import java.util.Arrays;

public class Classifier {

    ArrayList<Sample> trainingSamples;
    int numFeatures = 2;

    public Classifier() {
        trainingSamples = new ArrayList<>();
    }

    public Classifier(int numFeatures) {
        trainingSamples = new ArrayList<>();
        this.numFeatures = numFeatures;
    }

    public void addTrainingSample(double[] featureVector, int label) {
        addTrainingSample(new Sample(featureVector, label));
    }

    public void addTrainingSample(Sample sample) {
        trainingSamples.add(sample);
    }

    public void addTrainingSamples(Sample[] samples) {
        trainingSamples =  new ArrayList<Sample>(Arrays.asList(samples));
    }

    public void addTrainingSamples(ArrayList<Sample> samples) {
        trainingSamples.addAll(samples);
    }

    public void reset(){
        trainingSamples.clear();
    }

    public void train() {

    }

    public void setNumFeatures(int numFeatures){
        this.numFeatures = numFeatures;
    }

    double predict(Sample sample){
        return 0.0;
    }
}
