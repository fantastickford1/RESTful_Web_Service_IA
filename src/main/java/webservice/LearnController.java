package webservice;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import training.FaceClassification;
import training.Sample;

import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Scanner;

@RestController
public class LearnController {

    FaceClassification classification;

    @RequestMapping("/learn")
    public Learn learn(){
        int size = 100;
        this.classification = new FaceClassification(1728, size);
        try {
            this.classification.facesTraining();
        } catch (URISyntaxException e) {
            e.printStackTrace();
        }
        return new Learn("Aprendiendo");
    }

    @RequestMapping("/classify")
    public Learn result(@RequestParam(value="vector") String vector ){

        double[] confidence = new double[6];
        ///////////////////////////
        ArrayList<Float> hs = new ArrayList<>();
        Scanner scanner = new Scanner(vector);
        scanner.useLocale(Locale.US);
        while(scanner.hasNext()){
            if (scanner.hasNextFloat()){
                hs.add(scanner.nextFloat());
            }else if ( scanner.hasNextInt() ){
                hs.add((float)scanner.nextInt());
            }
        }
        float[] vect = new float[hs.size()];

        int i = 0;
        for (float sd :
                hs) {
            vect[i] = sd;
            i++;
        }

        Sample sample = new Sample(vect);
        int prediction = (int) this.classification.getSvmClassifier().predict(sample,confidence);
        return new Learn(FaceClassification.getKeysByValue(this.classification.getFaces(),prediction).iterator().next());
    }
}
