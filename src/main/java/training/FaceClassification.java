package training;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.HOGDescriptor;

import java.io.File;
import java.net.URISyntaxException;
import java.util.*;
import java.util.stream.Collectors;

public class FaceClassification {

    private final CascadeClassifier classifier;
    private final LibSVM svmClassifier;
    private final Map<String, Integer> faces = initFaces();
    private int size;

    private Map<String, Integer> initFaces() {

        Map<String, Integer> map = new HashMap<>();

        map.put("Enojado",      0);
        map.put("Sorpresa",     1);
        map.put("Miedo",        2);
        map.put("Felicidad",    3);
        map.put("Triste",       4);
        map.put("Disgusto",     5);

        return map;
    }

    public FaceClassification(int numFeatures, int size)
    {
       System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
       this.classifier = new CascadeClassifier(this.getPath(
               "/java/xml/haarcascade_frontalface_default.xml"
       ));
       this.svmClassifier = new LibSVM(numFeatures);
       this.size = size;
    }

    public Map<String, Integer> getFaces()
    {
        return Collections.unmodifiableMap(this.faces);
    }

    public String getPath(String url)
    {
        try {
            return new File(
                    getClass().getResource(url).toURI()
            ).getAbsolutePath();
        } catch (URISyntaxException ex) {
            System.err.println("Cant find file");
        }

        return null;
    }

    public void facesTraining() throws URISyntaxException {

        for (File folder : this.getFile("/image").listFiles()) {

            if (folder.isDirectory()) {

                String folderName = folder.getName();

                if (!folderName.equalsIgnoreCase("training")) {

                    int classNumber = this.faces.get(folderName);

                    System.out.println(folderName + " : " + classNumber + "\n");

                    for (File file : folder.listFiles()) {

                        Mat scarface = this.getScarface(
                                "/image/" + folderName + "/" + file.getName());

                        String extension = file.getName().split("\\.(?=[^\\.]+$)")[1];

                        if (extension.equalsIgnoreCase("jpg") ||
                                extension.equalsIgnoreCase("jpg") ||
                                extension.equalsIgnoreCase("jpeg")) {

                            if (scarface != null) {

                                System.out.println(file.getName() + " \t:\t " + scarface);

                                Mat imageResize = this.resize(scarface.clone(), new Size(size, size));
                                Mat grayImage = this.convertToGray(imageResize.clone());

                                float[] hog = this.getHOGDescriptors(grayImage).toArray();

                                System.out.println("size = " + hog.length);

                                Sample sample = new Sample(hog, classNumber);

                                this.svmClassifier.addTrainingSample(sample);

                            }
                        }

                    }
                }

            }

        }

        this.svmClassifier.train();

        System.out.println("OK Face");

    }

    public MatOfFloat getHOGDescriptors(Mat image) {

        Size winsize = new Size(64, 64);
        Size blocksize = new Size(32, 32);
        Size blockStride = new Size(16, 16);
        Size cellsize = new Size(16, 16);
        int nbins = 9;

        HOGDescriptor hog = new HOGDescriptor(
                winsize, blocksize, blockStride, cellsize, nbins);

        Size winStride = new Size(16, 16);
        Size padding = new Size(0, 0);

        MatOfFloat descriptors = new MatOfFloat();
        MatOfPoint locations = new MatOfPoint();

        hog.compute(image.clone(), descriptors, winStride, padding, locations);

        return descriptors;
    }

    public Mat resize(Mat image, Size size)
    {
        Mat dest = new Mat();
        Imgproc.resize(image, dest, size);
        return dest;
    }

    public Mat convertToGray(Mat mat)
    {
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY); return mat;
    }

    public LibSVM getSvmClassifier() {
        return svmClassifier;
    }

    public Mat getScarface(String fileName) {

        Mat image = Imgcodecs.imread(getPath(fileName));
        MatOfRect facesRect = new MatOfRect();

        this.classifier.detectMultiScale(image, facesRect);

        List<Rect> faces = new ArrayList<>(Arrays.asList(facesRect.toArray()));

        return !faces.isEmpty() ?
                new Mat(image, faces.iterator().next()) : null;

    }

    private File getFile(String path) throws URISyntaxException {
        return new File(getClass().getResource(path).toURI());
    }

    public static <T, E> Set<T> getKeysByValue(Map<T, E> map, E value) {
        return map.entrySet()
                .stream()
                .filter(entry -> Objects.equals(entry.getValue(), value))
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet());
    }
}
