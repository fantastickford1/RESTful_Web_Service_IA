package training;

public class Sample {

    private double[] featureVector;
    private int label;
    private int recordId;

    public Sample(float[] floatVector, int label){

        this.featureVector = new double[floatVector.length];

        for(int i = 0; i < floatVector.length; i++){
            this.featureVector[i] = (double) floatVector[i];
        }

        this.label = label;
    }

    public Sample(double[] featureVector, int label) {
        this.featureVector = featureVector;
        this.label = label;
    }

    public Sample(float[] floatVector){

        this.featureVector = new double[floatVector.length];

        for(int i = 0; i < floatVector.length; i++){
            this.featureVector[i] = (double)floatVector[i];
        }
    }

    public Sample(int featureVectorSize){
        featureVector = new double[featureVectorSize];
    }

    public double[] getFeatureVector() {
        return featureVector;
    }

    public void setFeatureVector(double[] featureVector) {
        this.featureVector = featureVector;
    }

    public int getLabel() {
        return label;
    }

    public int getRecordId() {
        return recordId;
    }

    public void setLabel(int label){
        this.label = label;
    }

    public void setRecordId(int recordId){
        this.recordId = recordId;
    }

}
