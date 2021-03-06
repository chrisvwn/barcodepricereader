package barcodepricereader;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

//import java.awt.FlowLayout;
//import java.awt.Image;
//import java.awt.image.BufferedImage;
//import java.awt.image.DataBufferByte;
//import java.awt.image.RenderedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

//import javax.imageio.ImageIO;
//import javax.swing.ImageIcon;
//import javax.swing.JFrame;
//import javax.swing.JLabel;

//import cern.colt.*;
//import cern.colt.matrix.DoubleMatrix2D;
//import cern.colt.matrix.linalg.Matrix2DMatrix2DFunction;

/**
 * Created by chris on 8/16/16.
 */
public class BarcodePriceReader1 {
    // The color of the outline drawn around the detected image.
    private final Scalar mLineColor = new Scalar(0, 255, 0);

    // You should have the trained data file in assets folder
    // You can get them at:
    // http://code.google.com/p/tesseract-ocr/downloads/list
    public static final String lang = "eng";

    //Methods
    public Mat resize(Mat img, int width, int height, int inter){

        inter = Imgproc.INTER_AREA;

        Size imgDim = img.size();

        Size dim = null;

        double r = 1;

        if(width <= 0 && height <= 0)
            return img;

        if (height == 0)
        {
            r =  width/imgDim.width;
            dim = new Size(width, (int)(img.height() * r));
        }
        else if(width == 0)
        {
            r = height/imgDim.height;
            dim = new Size((int)(img.width() * r), height);
        }
        else if (width > 0 && height > 0)
        {
            dim = new Size(width, height);
        }


        //resize the image
        Mat resized = new Mat();

        Imgproc.resize(img, resized, dim, 0, 0, inter);


        return resized;
    }

/*    public void displayImage(Image img2, String label)
    {
        //BufferedImage img=ImageIO.read(new File("/HelloOpenCV/lena.png"));
        ImageIcon icon=new ImageIcon(img2);

        JFrame frame=new JFrame(label);

        frame.setLayout(new FlowLayout());

        frame.setSize(img2.getWidth(null)+50, img2.getHeight(null)+50);

        JLabel lbl=new JLabel();

        lbl.setIcon(icon);

        frame.add(lbl);

        frame.setVisible(true);

        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
    }

    public Image toBufferedImage(Mat m){
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if ( m.channels() > 1 ) {
            Mat m2 = new Mat();
            Imgproc.cvtColor(m,m2,Imgproc.COLOR_BGR2RGB);
            type = BufferedImage.TYPE_3BYTE_BGR;
            m = m2;
        }
        byte [] b = new byte[m.channels()*m.cols()*m.rows()];
        m.get(0,0,b); // get all the pixels
        BufferedImage image = new BufferedImage(m.cols(),m.rows(), type);
        image.getRaster().setDataElements(0, 0, m.cols(),m.rows(), b);
        return image;

    }
*/
    public BarcodePriceReader1() throws IOException {
        //Constructor

    }

    public Mat detectMRZ(Mat img)
    {
        //Mat img = Imgcodecs.imread(photoPath);

        Mat roi = new Mat();

        int kernHt = img.height()/23;

        if (kernHt%2 != 0)
            kernHt++;

        Mat rectKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(13,5));

        Mat sqKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(21,kernHt));

        if (img.width() > 800)
            // load the image, resize it, and convert it to grayscale
            img = resize(img, 800, 600, Imgproc.INTER_AREA);

        //displayImage(toBufferedImage(img), "orig image resized");

        Mat gray = new Mat();

        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);

        //displayImage(toBufferedImage(gray), "image in grayscale");

        //smooth the image using a 3x3 Gaussian, then apply the blackhat
        //morphological operator to find dark regions on a light background
        Imgproc.GaussianBlur(gray, gray, new Size(3, 3), 0);

        //displayImage(toBufferedImage(gray), "gaussian blur");

        Mat blackhat = new Mat();
        Imgproc.morphologyEx(gray, blackhat, Imgproc.MORPH_BLACKHAT, rectKernel);

        //displayImage(toBufferedImage(blackhat), "blackhat");

        //compute the Scharr gradient of the blackhat image and scale the
        //result into the range [0, 255]
        Mat gradX = new Mat();
        //gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        Imgproc.Sobel(blackhat, gradX, CvType.CV_32F, 1, 0, -1, 1, 0);
        //gradX = Matrix absolute(gradX)

        //displayImage(toBufferedImage(gradX), "sobel");

        //(minVal, maxVal) = (np.min(gradX), np.max(gradX))
        MinMaxLocResult minMaxVal = Core.minMaxLoc(gradX);

        //gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        gradX.convertTo(gradX,CvType.CV_8U,255.0/(minMaxVal.maxVal-minMaxVal.minVal),-255.0/minMaxVal.minVal);

        //displayImage(toBufferedImage(gradX), "sobel converted to CV_8U");

        //apply a closing operation using the rectangular kernel to close
        //gaps in between letters -- then apply Otsu's thresholding method
        Imgproc.morphologyEx(gradX, gradX, Imgproc.MORPH_CLOSE, rectKernel);

        //displayImage(toBufferedImage(gradX), "closing operation morphology");

        Mat thresh = new Mat();
        Imgproc.threshold(gradX, thresh, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        //displayImage(toBufferedImage(thresh), "applied threshold");

        // perform another closing operation, this time using the square
        // kernel to close gaps between lines of the MRZ, then perform a
        // series of erosions to break apart connected components
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, sqKernel);

        //displayImage(toBufferedImage(thresh), "another closing operation morphology");

        Imgproc.erode(thresh, thresh, new Mat(), new Point(-1,-1), 4);

        //displayImage(toBufferedImage(thresh), "erode");
        // during thresholding, it's possible that border pixels were
        // included in the thresholding, so let's set 5% of the left and
        // right borders to zero
        int pRows = (int)(img.rows() * 0.05);
        int pCols = (int)(img.cols() * 0.05);

        //thresh[:, 0:pCols] = 0;
        //thresh.put(thresh.rows(), pCols, 0);
        //thresh[:, image.cols() - pCols] = 0;
        for (int i=0; i <= thresh.rows(); i++)
            for (int j=0; j<=pCols; j++)
                thresh.put(i, j, 0);

        //thresh[:, image.cols() - pCols] = 0;
        for (int i=0; i <= thresh.rows(); i++)
            for (int j=img.cols()-pCols; j<=img.cols(); j++)
                thresh.put(i, j, 0);

        //displayImage(toBufferedImage(thresh), "");

        // find contours in the thresholded image and sort them by their
        // size
        List<MatOfPoint> cnts = new ArrayList<MatOfPoint>();

        Imgproc.findContours(thresh.clone(), cnts, new Mat(), Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE);

        //cnts.sort(Imgproc.contourArea(contour));//, Imgproc.contourArea(cnts, true))

        // loop over the contours
        for (MatOfPoint c : cnts){
            // compute the bounding box of the contour and use the contour to
            // compute the aspect ratio and coverage ratio of the bounding box
            // width to the width of the image
            Rect bRect = Imgproc.boundingRect(c);
            int x=bRect.x;
            int y=bRect.y;
            int w=bRect.width;
            int h=bRect.height;

            int grWidth = gray.width();

            float ar = (float)w / (float)h;
            float crWidth = (float)w / (float)grWidth;

            // check to see if the aspect ratio and coverage width are within
            // acceptable criteria
            if (ar > 4 && crWidth > 0.75){
                // pad the bounding box since we applied erosions and now need
                // to re-grow it
                int pX = (int)((x + w) * 0.03); //previously 0.03 expanded to allow for warp
                int pY = (int)((y + h) * 0.03);
                x = x - pX;
                y = y - pY;
                w = w + (pX * 2);
                h = h + (pY * 2);

                // extract the ROI from the image and draw a bounding box
                // surrounding the MRZ
                //roi = new Mat(img, bRect);
                roi = new Mat(img, new Rect(x, y, w, h));

                Imgproc.rectangle(img, new Point(x, y), new Point(x + w, y + h), new Scalar(0, 255, 0), 2);

                //displayImage(toBufferedImage(img), "found mrz?");

                break;
            }
        }

        return roi;
    }

    private double angle(Point pt1, Point pt2, Point pt0)
    {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1*dx2 + dy1*dy2)/Math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
    }

    /* findSquares: returns sequence of squares detected on the image
     */
    void findSquares(Mat src, List<MatOfPoint> squares)
    {
        Mat src_gray = new Mat();
        Imgproc.cvtColor(src, src_gray, Imgproc.COLOR_BGR2GRAY);

        // Blur helps to decrease the amount of detected edges
        Mat filtered = new Mat();
        //Imgproc.blur(src_gray, filtered, new Size(3, 3));
        filtered = src_gray.clone();
        //displayImage(toBufferedImage(filtered), "blurred");

        // Detect edges
        Mat edges = new Mat();
        int thresh = 50;
        //Imgproc.Canny(filtered, edges, thresh, thresh*2);
        Imgproc.Canny(filtered, edges, thresh, 255);
        //displayImage(toBufferedImage(edges), "edges");


        // Dilate helps to connect nearby line segments
        Mat dilated_edges = new Mat();
        Imgproc.dilate(edges, dilated_edges, new Mat(), new Point(-1, -1), 2, 1, new Scalar(0,255,0)); // default 3x3 kernel
        //displayImage(toBufferedImage(dilated_edges), "dilated edges");

        // Find contours and store them in a list
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(dilated_edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        // Test contours and assemble squares out of them
        MatOfPoint2f approx = new MatOfPoint2f();

        approx.convertTo(approx, CvType.CV_32F);

        for (int i = 0; i < contours.size(); i++)
        {
            MatOfPoint cntCvt = new MatOfPoint();

            contours.get(i).convertTo(cntCvt, CvType.CV_32F);

            contours.set(i, cntCvt);

            // approximate contour with accuracy proportional to the contour perimeter
            Imgproc.approxPolyDP(new MatOfPoint2f(contours.get(i)), approx, Imgproc.arcLength(new MatOfPoint2f(contours.get(i)), true)*0.02, true);

            Point [] approx_array = new Point [4];

            approx_array = approx.toArray();
            MatOfPoint approx_matofpoint = new MatOfPoint(approx_array);

            // Note: absolute value of an area is used because
            // area may be positive or negative - in accordance with the
            // contour orientation

            if (approx_array.length == 4 && Math.abs(Imgproc.contourArea(new MatOfPoint2f(approx))) > 1000 &&
                    Imgproc.isContourConvex(approx_matofpoint))
            {
                double maxCosine = 0;
                for (int j = 2; j < 5; j++)
                {
                    double cosine = Math.abs(angle(approx_array[j%4], approx_array[j-2], approx_array[j-1]));
                    maxCosine = Math.max(maxCosine, cosine);
                }

                if (maxCosine < 0.3) {

                 //first sort the points into tl, tr, br, bl
                    int smallest = 0;
                    Point temp;

                    //bubble sort based on x
                    for (int idxPos = 0; idxPos < approx_array.length; idxPos++)
                        for (int idx = idxPos; idx < approx_array.length; idx++)
                        {
                            if (approx_array[idx].x < approx_array[idxPos].x)
                            {
                                temp = approx_array[idxPos];
                                approx_array[idxPos] = approx_array[idx];
                                approx_array[idx] = temp;
                            }
                        }

                    if (approx_array[1].y < approx_array[0].y)
                    {
                        temp = approx_array[0];
                        approx_array[0] = approx_array[1];
                        approx_array[1] = temp;
                    }

                    if (approx_array[3].y < approx_array[2].y)
                    {
                        temp = approx_array[2];
                        approx_array[2] = approx_array[3];
                        approx_array[3] = temp;
                    }

                    //order points clockwise from tl
                    MatOfPoint final_array = new MatOfPoint(approx_array[0], approx_array[2], approx_array[3], approx_array[1]);

                    squares.add(new MatOfPoint(final_array));
                }
            }
        }
    }

    /* findLargestSquare: find the largest square within a set of squares
     */
    private MatOfPoint findLargestSquare(List<MatOfPoint> squares)
    {
        if (squares.size() == 0)
        {
            //std::cout << "findLargestSquare !!! No squares detect, nothing to do." << std::endl;
            return new MatOfPoint();
        }

        int max_width = 0;
        int max_height = 0;
        int max_square_idx = 0;

        for (int i = 0; i < squares.size(); i++)
        {
            // Convert a set of 4 unordered Points into a meaningful cv::Rect structure.
            Rect rectangle = Imgproc.boundingRect(squares.get(i));

            //std::cout << "find_largest_square: #" << i << " rectangle x:" << rectangle.x << " y:" << rectangle.y << " " << rectangle.width << "x" << rectangle.height << endl;

            // Store the index position of the biggest square found
            if ((rectangle.width >= max_width) && (rectangle.height >= max_height))
            {
                max_width = rectangle.width;
                max_height = rectangle.height;
                max_square_idx = i;
            }
        }

        return squares.get(max_square_idx);
    }

    private Mat findRoundedCornersID(Mat src)
    {
        if (src.empty())
        {
            return src;
        }

        if (src.width() > 800)
            while (src.width() > 800)// || img.height() > 600)
                Imgproc.pyrDown(src, src);

        List<MatOfPoint> squares = new ArrayList<MatOfPoint>();

        findSquares(src, squares);

        if (squares.size() == 0)
            return src;

        // Draw all detected squares
        Mat src_squares = src.clone();
        for (int i = 0; i < squares.size(); i++)
        {
            int n = squares.get(i).rows();
            Imgproc.polylines(src_squares,squares, true, new Scalar(255, 0, 0), 2, Core.LINE_AA, 0);
        }

        //displayImage(toBufferedImage(src_squares), "src squares");

        MatOfPoint largest_square = null;
        largest_square = findLargestSquare(squares);

        if (largest_square == null)
            return src;

        List<Point> largest_square_list = new ArrayList<Point>();
        largest_square_list = largest_square.toList();

        Point [] sqPts = largest_square.toArray();

        double wSq1 = sqPts[1].x - sqPts[0].x;
        double wSq2 = sqPts[2].x - sqPts[3].x;

        double hSq1 = sqPts[3].y - sqPts[0].y;
        double hSq2 = sqPts[2].y - sqPts[1].y;

        double sqRatio = 0.03;

        Point sqPt0 = new Point(sqPts[0].x - wSq1*sqRatio, sqPts[0].y - hSq1*sqRatio);
        Point sqPt1 = new Point(sqPts[1].x + wSq1*sqRatio, sqPts[1].y - hSq2*sqRatio);
        Point sqPt2 = new Point(sqPts[2].x + wSq2*sqRatio, sqPts[2].y + hSq2*sqRatio);
        Point sqPt3 = new Point(sqPts[3].x - wSq2*sqRatio, sqPts[3].y + hSq1*sqRatio);

        //MatOfPoint2f inputQuad = new MatOfPoint2f(sqPt0,sqPt1,sqPt2,sqPt3);

        MatOfPoint2f inputQuad = new MatOfPoint2f(
                new Point( sqPts[0].x, sqPts[0].y ),
                new Point( sqPts[1].x, sqPts[1].y ),
                new Point( sqPts[2].x, sqPts[2].y ),
                new Point( sqPts[3].x, sqPts[3].y ));


        MatOfPoint inputQuadPt = new MatOfPoint(
                new Point( sqPts[0].x, sqPts[0].y ),
                new Point( sqPts[1].x, sqPts[1].y ),
                 new Point( sqPts[2].x, sqPts[2].y ),
                new Point( sqPts[3].x, sqPts[3].y ));

        //okay. found largest square. let's warp it to a real rectangle

        //get the bounding rectangle of the largest square
        Rect bRect = Imgproc.boundingRect(largest_square);

        //get top left and width and height in preparation to calculate 4 corner points of the bounding rect
        int x=bRect.x;
        int y=bRect.y;
        int w=bRect.width;
        int h=bRect.height;


        //we need it in MatofPoint2f format for getPerspectiveTransform
        MatOfPoint2f outputQuad = new MatOfPoint2f(
                new Point( x, y ),
                new Point( x+w, y),
                new Point( x+w, y+h),
                new Point( x, y+h));

        inputQuad.convertTo(inputQuad, CvType.CV_32F);
        outputQuad.convertTo(outputQuad, CvType.CV_32F);
        Mat lambda = Imgproc.getPerspectiveTransform( inputQuad, outputQuad );

        Mat warpSrc = new Mat(src,bRect);

        // Apply the Perspective Transform just found to the src image
        //Imgproc.warpAffine(src.submat(bRect), warpSrc, lambda, bRect.size(), Imgproc.INTER_CUBIC);

        //display orig and warped largest square
        Imgproc.warpPerspective(src.submat(bRect), warpSrc, lambda, src.size(), Imgproc.INTER_CUBIC);

        src = warpSrc;

        Imgproc.line(src,sqPts[0],sqPts[1], new Scalar(0,0,255),1,Imgproc.LINE_AA,0);
        Imgproc.line(src,sqPts[1],sqPts[2], new Scalar(0,0,255),1,Imgproc.LINE_AA,0);
        Imgproc.line(src,sqPts[2],sqPts[3], new Scalar(0,0,255),1,Imgproc.LINE_AA,0);
        Imgproc.line(src,sqPts[3],sqPts[0], new Scalar(0,0,255),1,Imgproc.LINE_AA,0);

        Imgproc.rectangle(src,bRect.tl(),bRect.br(), new Scalar(0,255,0),1);

        //now that we have corrected the image find the squares again

        findSquares(src, squares);

        if (squares.size() == 0)
            return src;

        largest_square = findLargestSquare(squares);

        if (largest_square == null)
            return src;

        largest_square_list = largest_square.toList();

        // Draw circles at the corners
        for (int i = 0; i < largest_square_list.size(); i++)
            Imgproc.circle(src, largest_square_list.get(i), 4, new Scalar(255, 255, 224), Core.FILLED);

        bRect = Imgproc.boundingRect(largest_square);

        warpSrc = new Mat(bRect.size(), 0);


        for (int i = 0; i < 4/*outputQuad.toArray().length*/; i++ )
            Imgproc.circle(src, outputQuad.toArray()[i], 4, new Scalar(255, 255, 224), Core.FILLED);

        //displayImage(toBufferedImage(src), "corners");

//        src = new Mat(src, bRect);
        src = warpSrc;

        //displayImage(toBufferedImage(idInSrc), "cropped to id only");

        return src;
    }
    //static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public void run(String inFile, String outFile) {
        System.out.println("\nRunning MRZ search ...");

        Mat img = Imgcodecs.imread(inFile);

        Mat detectedID = findRoundedCornersID(img);

        //write detectedID image to storage
        Imgcodecs.imwrite("/home/chris/Documents/detectedID.png", detectedID);


        Mat mrz = detectMRZ(detectedID);

        //displayImage(toBufferedImage(mrz), "found MRZ?");

        if (mrz.empty()) {
        	System.out.println("Nothing found");
            return;
        }

        //write mrz image to storage
        Imgcodecs.imwrite(outFile, mrz);

/*        //TESS
        TessBaseAPI baseApi = new TessBaseAPI();
        baseApi.setVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890<");
        baseApi.setDebug(true);
        baseApi.init(DATA_PATH, "eng");

        Mat result = new Mat();
        Imgproc.cvtColor(mrz, result, Imgproc.COLOR_RGB2BGRA);

        //Imgproc.cvtColor(result, result, Imgproc.COLOR_BGRA2GRAY);

        //Imgproc.GaussianBlur(result, result, new Size(3, 3), 0);

        //Imgproc.threshold(result, result, 0, 255, Imgproc.THRESH_OTSU);

        Bitmap bmp = Bitmap.createBitmap(result.cols(), result.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(result, bmp);

        baseApi.setImage(bmp);
        final String recognizedText = baseApi.getUTF8Text();
        recognizedText.toUpperCase();
        recognizedText.replaceAll(" ", "");
        Pix tess_thresh_img = baseApi.getThresholdedImage();
        baseApi.end();

        File pixFile = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_PICTURES).toString()+"/SecondSight/tess_img_"+
                System.currentTimeMillis()+".jpg");

        WriteFile.writeImpliedFormat(tess_thresh_img,pixFile);

        //Imgproc.putText(img, recognizedText, new Point(0, img.height()-20), 1, 1.0, new Scalar(0, 255, 0), 2);

        if (recognizedText.equals(""))
        {
            thisActivity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(thisContext, "No text recognized",
                            Toast.LENGTH_SHORT).show();
                }
            });
        }
        else
        {
            thisActivity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(thisContext, "OCR successful:\n"+ recognizedText,
                            Toast.LENGTH_LONG).show();
                }
            });
        }
        return recognizedText;
*/
    }

//    public static void main(String[] args) {
//        System.out.println(args[0]);
//
//        try{
//            new BarcodePriceReader1().run(args[0], args[1]);
//        }
//        catch(Exception e)
//        {
//            e.printStackTrace();
//        }
//    }
}