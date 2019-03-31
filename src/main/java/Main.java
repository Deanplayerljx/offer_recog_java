import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
//import net.sourceforge.tess4j.Tesseract;
import java.io.File;
import static org.opencv.imgproc.Imgproc.INTER_AREA;
import net.sourceforge.tess4j.*;
public class Main {

    static final String imageFolderPath="iimage/";
    static final String templatePath = "template5.jpg";
    static final String[] imageNames = {"test1.jpeg","test2.jpeg","test3.jpg","test4.jpg","test5.jpg","test6.jpg","test_hard1.jpg","test_hard2.jpg","test_hard3.jpeg"};
    public static void main(String[] args) {

        ITesseract instance = new Tesseract();
        instance.setDatapath("/Users/DeanLin/Downloads/Tess4J/tessdata");
        instance.setLanguage("eng");
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        for (String imageName : imageNames) {
            Mat image = Imgcodecs.imread(imageFolderPath + imageName, 0);
            Mat template = Imgcodecs.imread(templatePath, 0);
            Mat edges = new Mat(template.height(), template.width(), 0);
            Imgproc.Canny(template, edges, 180,185);

            String name = matchTemplate(edges, image, imageName.split("\\.",2)[0] + "result.jpg", instance);
            if (name == null) {
                System.out.println("invalid offer");
            }
            else {
                System.out.println(name);
            }
        }
    }

    private static String matchTemplate(Mat template, Mat image, String outFileName, ITesseract instance) {
        int templateHeight = template.height();
        int templateWidth = template.width();
        boolean found = false;
        Core.MinMaxLocResult bestResult = null;
        double bestRatio = 0;
        Mat matchedTemplate = new Mat();
        Imgproc.matchTemplate(image, template, matchedTemplate, Imgproc.TM_CCOEFF);
        double start = 0.2;
        double end = 1;
        double step = 0.04;
        for (double scale = end; scale > start; scale -= step) {

            Mat resized = new Mat();
            Size size = new Size(image.width()*scale, image.height()*scale);

            Imgproc.resize(image, resized, size,0,0,INTER_AREA);
            Imgproc.Canny(resized, resized, 180,185);
            double ratio = (double)(image.height()) / resized.height();

            if (resized.height() < template.height() || resized.width() < template.width()) {
                break;
            }

            Mat res = new Mat();
            Imgproc.matchTemplate(resized, template, res, Imgproc.TM_CCOEFF);
            Core.MinMaxLocResult result = Core.minMaxLoc(res);
            if (found == false || result.maxVal > bestResult.maxVal) {
                bestResult = result;
                bestRatio = ratio;
                found = true;
            }
        }

        int startX = (int) (bestResult.maxLoc.x * bestRatio);
        int startY = (int) (bestResult.maxLoc.y * bestRatio);
        int endX = (int) ((bestResult.maxLoc.x + templateWidth)*bestRatio);
        int endY = (int) ((bestResult.maxLoc.y + templateHeight)*bestRatio);

        Imgproc.rectangle(image, new Point(startX, startY), new Point(endX, endY), new Scalar(255, 0, 0));
        Mat cropped = image.submat(startY, endY,startX, endX);

        Imgcodecs.imwrite(outFileName, cropped);
        File imageFile = new File(outFileName);
        try {
            String result = instance.doOCR(imageFile);
            result = result.replaceAll("[\\s'!]","").toLowerCase();
            String[] splitted = result.split(",");
            if (splitted.length >= 2) {
                if (splitted[1].contains("anill")) {
                    String name = splitted[0];
                    return name;
                }
            }
            return null;
        } catch (TesseractException e) {
            System.err.println(e.getMessage());
        }
        return null;
    }
}
