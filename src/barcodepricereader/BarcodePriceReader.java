package barcodepricereader;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;

import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.math.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import cern.colt.*;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Matrix2DMatrix2DFunction;

import com.google.zxing.*;
import com.google.zxing.client.j2se.BufferedImageLuminanceSource;
import com.google.zxing.common.HybridBinarizer;

import net.sourceforge.tess4j.*;

class BarcodePriceReader {
    // The color of the outline drawn around the detected image.
    private final Scalar mLineColor = new Scalar(0, 255, 0);

	
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
	
	public void displayImage(Mat img, String label)
	{   
		//BufferedImage img=ImageIO.read(new File("/HelloOpenCV/lena.png"));
		Mat tmp = img.clone();
				
		if (tmp.width() > 800 || tmp.height() > 800)
            while (tmp.width() > 800 || tmp.height() > 800)
                Imgproc.pyrDown(tmp, tmp);
		
		Image img2 = toBufferedImage(tmp);
		
		tmp = null;
		
		ImageIcon icon=new ImageIcon(img2);
		
		JFrame frame=new JFrame(label);
		
		frame.setLayout(new FlowLayout());        
		
		frame.setSize(img2.getWidth(null)+50, img2.getHeight(null)+50);     
		
		JLabel lbl=new JLabel();
		
		lbl.setIcon(icon);
		
		frame.add(lbl);
		
		frame.setVisible(true);
		
		frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
		
		try{
			Thread.sleep(3000);
		}
			catch(Exception e){
		}
		
		frame.setVisible(false);
		
		frame.dispose();
		
		img2 = null;
		
		icon = null;
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
	
	public BarcodePriceReader() throws IOException {
	}
	
	public Mat detectIDMRZ(Mat img)
	{
        Mat roi = null;

        Mat rectKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(13,5));

        Mat sqKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(21,21));
        
        if (img.width() > 800)
	        // load the image, resize it, and convert it to grayscale
	        img = resize(img, 800, 600, Imgproc.INTER_AREA);
        
        //displayImage(img, "orig image resized");

        Mat gray = new Mat();
        
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);      
        
        //displayImage(gray, "image in grayscale");

        //Detect barcode
      //smooth the image using a 3x3 Gaussian, then apply the blackhat
    	//morphological operator to find dark regions on a light background
    	Imgproc.GaussianBlur(gray, gray, new Size(3, 3), 0);
    	
    	//displayImage(gray, "gaussian blur");
    	
    	Mat blackhat = new Mat();
    	Imgproc.morphologyEx(gray, gray, Imgproc.MORPH_BLACKHAT, rectKernel);
        
    	//displayImage(blackhat, "blackhat");
    	
    	//Threshold the image
    	Imgproc.threshold(blackhat, blackhat, 250, 255, Imgproc.THRESH_BINARY);
    	
        // compute the Scharr gradient magnitude representation of the images
        // in both the x and y direction
    	Mat gradX = new Mat();
    	//gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    	Imgproc.Sobel(gray, gradX, CvType.CV_32F, 1, 0, -1, 1, 0);
    	
    	Mat gradY = new Mat();
    	//gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    	Imgproc.Sobel(gray, gradY, CvType.CV_32F, 0, 1, -1, 1, 0);
    	
    	gray = null;
    	
    	Mat grad = new Mat();
    	
    	Core.subtract(gradX, gradY, grad);
    	Core.convertScaleAbs(grad, grad);
    	
    	gradX = null;
    	gradY = null;
    	
    	//displayImage(grad, "sobel scharr gradient");
    	
    	//smooth the image using a 3x3 Gaussian, then apply the blackhat
    	//morphological operator to find dark regions on a light background
    	Imgproc.GaussianBlur(grad, grad, new Size(3, 3), 0);
    	
    	//displayImage(grad, "gaussian blur");
    	
    	//Threshold the image
    	Mat thresh = new Mat();
    	Imgproc.threshold(grad, thresh, 250, 255, Imgproc.THRESH_BINARY);
    	
    	grad = null;
    	
    	//displayImage(grad, "threshold");
    	
    	//construct a closing kernel and apply it to the thresholded image
    	Mat closed = new Mat();
    	Imgproc.morphologyEx(thresh, closed, Imgproc.MORPH_CLOSE, rectKernel);
    	
    	//displayImage(closed, "close");
    	
    	thresh = null;
    	
    	//perform a series of erosions and dilations
    	Imgproc.erode(closed, closed, new Mat(), new Point(-1,-1), 4);
    	
    	Imgproc.dilate(closed, closed, new Mat(), new Point(-1,-1), 4);
    	
    	//displayImage(closed, "erode");
    	
    	// find contours in the thresholded image and sort them by their
    	// size
    	List<MatOfPoint>cnts = new ArrayList<MatOfPoint>(); 
    	
    	Imgproc.findContours(closed.clone(), cnts, new Mat(), Imgproc.RETR_EXTERNAL,
        		Imgproc.CHAIN_APPROX_SIMPLE);
    	
    	closed = null;
    	
    	Collections.sort(cnts,  new Comparator<MatOfPoint>() {
    		@Override
    	    public int compare(MatOfPoint cnt1, MatOfPoint cnt2) {
    			Double a1 = Imgproc.contourArea(cnt1, true);
    			Double a2 = Imgproc.contourArea(cnt2, true);

    	        return a1.compareTo(a2);
    	    }
    	});
    	
    	Rect bRect = Imgproc.boundingRect(cnts.get(cnts.size()-1));
		int x=bRect.x;
		int y=bRect.y;
		int w=bRect.width;
		int h=bRect.height;
		
		roi = new Mat(img, new Rect(x, y, w, h));
		
		//Imgproc.rectangle(img, new Point(x, y), new Point(x + w, y + h), new Scalar(0, 0, 255), 2);
		
		//displayImage(img, "found mrz?");
		
    	return roi;
    	
	}
	
    //IMPORTED
    
    public Mat detectMRZ(Mat img)
    {
        //Mat img = Imgcodecs.imread(photoPath);

        Mat roi = new Mat();

        int kernHt = img.height()/23;

        if (kernHt%2 != 0)
            kernHt++;

        Mat rectKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(13,5));

        Mat sqKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(21,kernHt));

        if (img.height() > 800)
            // load the image, resize it, and convert it to grayscale
            img = resize(img, 800, 600, Imgproc.INTER_AREA);

        //displayImage(img, "orig image resized");

        Mat gray = img;

        //Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);

        //displayImage(gray, "image in grayscale");

        //smooth the image using a 3x3 Gaussian, then apply the blackhat
        //morphological operator to find dark regions on a light background
        Imgproc.GaussianBlur(gray, gray, new Size(3, 3), 0);

        //displayImage(gray, "gaussian blur");

        Mat blackhat = new Mat();
        Imgproc.morphologyEx(gray, blackhat, Imgproc.MORPH_BLACKHAT, rectKernel);

        //displayImage(blackhat, "blackhat");

        gray = null;
        
        //compute the Scharr gradient of the blackhat image and scale the
        //result into the range [0, 255]
        Mat gradX = new Mat();
        //gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        Imgproc.Sobel(blackhat, gradX, CvType.CV_32F, 1, 0, -1, 1, 0);
        //gradX = Matrix absolute(gradX)

        //displayImage(gradX, "sobel");

        blackhat = null;
        
        //(minVal, maxVal) = (np.min(gradX), np.max(gradX))
        MinMaxLocResult minMaxVal = Core.minMaxLoc(gradX);

        //gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        gradX.convertTo(gradX,CvType.CV_8U,255.0/(minMaxVal.maxVal-minMaxVal.minVal),-255.0/minMaxVal.minVal);

        //displayImage(gradX, "sobel converted to CV_8U");

        //apply a closing operation using the rectangular kernel to close
        //gaps in between letters -- then apply Otsu's thresholding method
        Imgproc.morphologyEx(gradX, gradX, Imgproc.MORPH_CLOSE, rectKernel);

        //displayImage(gradX, "closing operation morphology");

        Mat thresh = new Mat();
        Imgproc.threshold(gradX, thresh, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        gradX = null;
        
        //displayImage(thresh, "applied threshold");

        // perform another closing operation, this time using the square
        // kernel to close gaps between lines of the MRZ, then perform a
        // series of erosions to break apart connected components
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, sqKernel);

        //displayImage(thresh, "another closing operation morphology");

        Imgproc.erode(thresh, thresh, new Mat(), new Point(-1,-1), 4);

        //displayImage(thresh, "erode");
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

        //displayImage(thresh, "");

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
            //if (ar > 4 && crWidth > 0.75){
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

                //displayImage(img, "found mrz?");

                //break;
            //}
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
//    	Mat imgHSV = new Mat();
//        Imgproc.cvtColor(src, imgHSV, Imgproc.COLOR_BGR2HSV);
//        
//        Mat imgThreshed = new Mat();
//        
//        Core.inRange(imgHSV, new Scalar(30, 70, 85), new Scalar(50, 255, 255), imgThreshed);
//
//        Mat tmp = imgThreshed.clone();
//        
//        displayImage(tmp, "imgThreshed");
//        
//        tmp = null;

        //Mat src_gray = new Mat();
        //Imgproc.cvtColor(imgThreshed, src_gray, Imgproc.COLOR_HSV2BGR);
        
        //Imgproc.cvtColor(imgThreshed, src_gray, Imgproc.COLOR_BGR2GRAY);
        
        // Blur helps to decrease the amount of detected edges
    	
    	Mat rectKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(9,1));

        Mat sqKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(21,21));
    	
    	Mat gray = new Mat();
    	
    	Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);      
        
        //displayImage(gray, "image in grayscale");
        
    	//smooth the image using a 3x3 Gaussian, then apply the blackhat
    	//morphological operator to find dark regions on a light background
    	Imgproc.GaussianBlur(gray, gray, new Size(3, 3), 0);
    	
    	//displayImage(gray, "gaussian blur");
    	
    	Imgproc.morphologyEx(gray, gray, Imgproc.MORPH_BLACKHAT, rectKernel);
        
    	//displayImage(gray, "blackhat");
    	
    	//Threshold the image
    	//Imgproc.adaptiveThreshold(gray, gray, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY,15,-5);
    	
    	//displayImage(gray, "adaptive thresh");
    	
    	//compute the Scharr gradient of the blackhat image and scale the
    	//result into the range [0, 255]
    	Mat gradX = new Mat();
    	//gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    	//Imgproc.Sobel(gray, gradX, CvType.CV_32F, 1, 0, -1, 1, 0);
    	//Core.convertScaleAbs(gradX, gradX);
    	
    	//displayImage(gradX, "sobel");
    	
        Mat filtered = new Mat();

        filtered = gray.clone();
        
    	//perform a series of erosions and dilations
    	//Imgproc.erode(filtered, filtered, new Mat(), new Point(-1,-1), 2);
    	
    	Imgproc.dilate(filtered, filtered, new Mat(), new Point(-1,-1), 10);
        
        //displayImage(filtered, "erode/dilate");

        // Detect edges
        Mat edges = new Mat();
        int thresh = 250;
        //Imgproc.Canny(filtered, edges, thresh, thresh*2);
        Imgproc.Canny(filtered, edges, 50, 255);
        //displayImage(edges, "edges");

        filtered = null;
        
        // Dilate helps to connect nearby line segments
        Mat dilated_edges = new Mat();
        Imgproc.dilate(edges, dilated_edges, new Mat(), new Point(-1, -1), 2, 1, new Scalar(0,255,0)); // default 3x3 kernel
        //displayImage(dilated_edges, "dilated edges");

        Imgproc.dilate(dilated_edges, dilated_edges, new Mat(), new Point(-1,-1), 20);
        
        //displayImage(dilated_edges, "dilated edges");
        
        Imgproc.Canny(dilated_edges, dilated_edges, 250, 255);
        //displayImage(dilated_edges, "edges2");
        
        edges = null;
        
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

            if(approx_array.length == 4 )
            {
            if (Math.abs(Imgproc.contourArea(new MatOfPoint2f(approx))) > src.height()*src.width()*0.10 &&
                    Imgproc.isContourConvex(approx_matofpoint))
            {
                double maxCosine = 0;
                for (int j = 2; j < 5; j++)
                {
                    double cosine = Math.abs(angle(approx_array[j%4], approx_array[j-2], approx_array[j-1]));
                    maxCosine = Math.max(maxCosine, cosine);
                }

                if (maxCosine < 0.4) {

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
            else
            {
            	RotatedRect rect = Imgproc.minAreaRect(approx);
            	
                Point [] rect_array = new Point[4];
                rect.points(rect_array);
                
                // matrices we'll use
                Mat M;
                Mat cropped = new Mat();
                Mat rotated = new Mat();
                
                // get angle and size from the bounding box
                double angle = rect.angle;
                Size rect_size = rect.size;
                
                // thanks to http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
                if (rect.angle < -45.) {
                    angle += 90.0;
                    double temp = rect_size.width;
                    rect_size.width = rect_size.height;
                    rect_size.height = temp;
                }
                
                // get the rotation matrix
                M = Imgproc.getRotationMatrix2D(rect.center, angle, 1.0);
                
                // perform the affine transformation
                Imgproc.warpAffine(src, rotated, M, src.size(), Imgproc.INTER_CUBIC);
                
                // crop the resulting image
                Imgproc.getRectSubPix(rotated, rect_size, rect.center, cropped);
                
                //if the area of the contour rotated if necessary is > 10000
                //and is convex
                if (cropped.height()*cropped.width() > src.height()*src.width()*0.10 &&
                        !Imgproc.isContourConvex(approx_matofpoint))
            	{
                	Mat tmp = src.clone();
                	Imgproc.rectangle(tmp, rect_array[0], rect_array[2], new Scalar(255,255,255), 5);
                	
                	//displayImage(tmp, "min area rect");
                	
                    //displayImage(cropped, "contour points > 4");

                	for(int j=0; j < rect_array.length; j++)
                	{
                		if(rect_array[j].x < 0)
                			rect_array[j].x = 0;
                		if(rect_array[j].x > src.width())
                			rect_array[j].x = src.width()-1;
                		if(rect_array[j].y < 0)
                			rect_array[j].y = 0;
                		if(rect_array[j].y > src.height())
                			rect_array[j].y = src.height()-1;
                	}
                	
                    MatOfPoint final_array = new MatOfPoint(rect_array[2], rect_array[3], rect_array[0], rect_array[1]);

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

    private List<MatOfPoint> findLabel(Mat src)
    {
        if (src.empty())
        {
            //return src;
        	return(null);
        }

        int maxLen = 800;
        
        if(src.width() > src.height())
        {
            int ratio = src.width()/maxLen;
        	if (src.width() > maxLen)
       			// load the image, resize it, and convert it to grayscale
       			src = resize(src, maxLen, src.height()/ratio, Imgproc.INTER_AREA);
        }
        else
        {
            int ratio = src.height()/maxLen;
            if (src.height() > maxLen)
            	// load the image, resize it, and convert it to grayscale
            	src = resize(src, src.height()/ratio, maxLen, Imgproc.INTER_AREA);        	
        }
        
        List<MatOfPoint> squares = new ArrayList<MatOfPoint>();

        findSquares(src, squares);

        if (squares.size() == 0)
            return null;

        // Draw all detected squares
        Mat src_squares = src.clone();
        for (int i = 0; i < squares.size(); i++)
        {
            int n = squares.get(i).rows();
            Imgproc.polylines(src_squares,squares, true, new Scalar(255, 0, 0), 5, Core.LINE_AA, 0);
        }

        MatOfPoint largest_square = null;
        largest_square = findLargestSquare(squares);

        if (largest_square == null)
            return null;
/*
        if(Imgproc.contourArea(largest_square) > src.height()*src.width()*0.10)
        {
        List<Point> largest_square_list = new ArrayList<Point>();
        largest_square_list = largest_square.toList();

        Point [] sqPts = largest_square.toArray();

        Imgproc.rectangle(src_squares, sqPts[0], sqPts[2], new Scalar(255,255,255),10);
        displayImage(src_squares, "src squares2");
        
        double wSq1 = sqPts[1].x - sqPts[0].x;
        double wSq2 = sqPts[2].x - sqPts[3].x;

        double hSq1 = sqPts[3].y - sqPts[0].y;
        double hSq2 = sqPts[2].y - sqPts[1].y;

        double sqRatio = 0.03;

        Point sqPt0 = new Point(sqPts[0].x - wSq1*sqRatio, sqPts[0].y - hSq1*sqRatio);
        Point sqPt1 = new Point(sqPts[1].x + wSq1*sqRatio, sqPts[1].y - hSq2*sqRatio);
        Point sqPt2 = new Point(sqPts[2].x + wSq2*sqRatio, sqPts[2].y + hSq2*sqRatio);
        Point sqPt3 = new Point(sqPts[3].x - wSq2*sqRatio, sqPts[3].y + hSq1*sqRatio);

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

		if(bRect.tl().x < 0)
			bRect.tl().x = 0;
		if(bRect.br().x > src.width())
			bRect.br().x = src.width();
		if(bRect.tl().y < 0)
			bRect.tl().y = 0;
		if(bRect.br().y > src.height())
			bRect.br().y = src.height();
        
        Mat warpSrc = new Mat(src,bRect);
        //Imgproc.getRectSubPix(src, bRect.size(), new Point((bRect.x+bRect.width)/2,(bRect.y+bRect.height)/2), warpSrc);

       //display orig and warped largest square
       Imgproc.warpPerspective(src, warpSrc, lambda, src.size(), Imgproc.INTER_CUBIC);

        src = warpSrc;
        
        Mat src_clone = src.clone();
        Imgproc.line(src_clone,sqPts[0],sqPts[1], new Scalar(0,0,255),10,Imgproc.LINE_AA,0);
        Imgproc.line(src_clone,sqPts[1],sqPts[2], new Scalar(0,0,255),10,Imgproc.LINE_AA,0);
        Imgproc.line(src_clone,sqPts[2],sqPts[3], new Scalar(0,0,255),10,Imgproc.LINE_AA,0);
        Imgproc.line(src_clone,sqPts[3],sqPts[0], new Scalar(0,0,255),10,Imgproc.LINE_AA,0);

        Imgproc.rectangle(src_clone,bRect.tl(),bRect.br(), new Scalar(0,0,255),10);

        displayImage(src_clone, "warp src");
        
        src_clone=null;
        
        //now that we have corrected the image find the squares again

        findSquares(src, squares);

        if (squares.size() == 0)
            return src;

        largest_square = findLargestSquare(squares);

        if (largest_square == null)
            return src;

        largest_square_list = largest_square.toList();

        // Draw circles at the corners
        Mat src_clone = src.clone();
        
        for (int i = 0; i < largest_square_list.size(); i++)
            Imgproc.circle(src_clone, largest_square_list.get(i), 10, new Scalar(255, 255, 224), Core.FILLED);

        displayImage(src_clone, "largest square corners");
        
        src_clone = null;
        
        Rect bRect = Imgproc.boundingRect(largest_square);

        double trimPerc = 0.05;
        
        //bRect.x = (int)Math.round(bRect.x + bRect.width * trimPerc);
        //bRect.y = (int)Math.round(bRect.y + bRect.height * trimPerc);
        //bRect.width = (int)Math.round(bRect.width - bRect.width * trimPerc*2);
        //bRect.height = (int)Math.round(bRect.height - bRect.height * trimPerc*2);
        
        Mat warpSrc = src.submat(bRect);
        //Imgproc.getRectSubPix(src, bRect.size(), new Point((bRect.x+bRect.width)/2,(bRect.y+bRect.height)/2), warpSrc);

        src_clone = src.clone();
        for (int i = 0; i < 4; i++ )
            Imgproc.circle(src_clone, outputQuad.toArray()[i], 10, new Scalar(255, 255, 224), Core.FILLED);

        displayImage(src_clone, "corners");

//        src = new Mat(src, bRect);
        src = warpSrc;
        } //end if largest_square area > 1000
*/

        Rect bRect = Imgproc.boundingRect(largest_square);
        
        //src = src.submat(bRect);

        //displayImage(src, "cropped to id only");

        //return src;
        return squares;
    }
    
    //END IMPORTED
    
    static String opencvpath = System.getProperty("user.dir") + "\\";
    static{String libPath = System.getProperty("java.library.path");}
    static String cvPath = opencvpath + Core.NATIVE_LIBRARY_NAME + ".dll";
    static{System.load(cvPath);}
    
	//static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
	
    public void run(String inFile) {
        //System.out.println("\nRunning MRZ search ...");
            	    	
        Mat img = Imgcodecs.imread(inFile);

        Mat temp = img.clone();
        
        //displayImage(temp, "orig image scaled");
        
        temp = null;
        
        //IMPORTED
        
        //Mat detectedID = findLabel(img);
        
        List<MatOfPoint> squares = findLabel(img);

        //temp = detectedID.clone();
        
        //displayImage(temp, "detected label");
        
        temp = null;
        
        //write detectedID image to storage
        //Imgcodecs.imwrite("detectedID.png", detectedID);

        //Mat roi = detectMRZ(detectedID);
        
        //END IMPORTED
		
        //Mat roi = detectIDMRZ(detectedID);
        
        //System.out.println("Barcode text is " + result.getText());
        
        //displayImage(roi, "found MRZ?");
        //Imgcodecs.imwrite("/home/chris/Documents/mrz_roi.jpg", roi);

        Mat src = img.clone();
        
        int maxLen = 800;
        
        if(img.width() > src.height())
        {
            int ratio = src.width()/maxLen;
        	if (src.width() > maxLen)
       			// load the image, resize it, and convert it to grayscale
       			src = resize(src, maxLen, src.height()/ratio, Imgproc.INTER_AREA);
        }
        else
        {
            int ratio = src.height()/maxLen;
            if (src.height() > maxLen)
            	// load the image, resize it, and convert it to grayscale
            	src = resize(src, src.height()/ratio, maxLen, Imgproc.INTER_AREA);        	
        }
        
        int i = 0;
        Result barcodeResult = null;
        String tessResult = null;
        Mat detectedLabel;
        Rect bRect;
        LuminanceSource source;
        BinaryBitmap bitmap;
        Reader reader;
        ITesseract tessInstance = new Tesseract();

        
        if(squares == null)
        {
        	displayImage(src, "No squares found");
        	
        	System.out.println("Barcode not found.");
        }
        else
        while(i < squares.size())
        {
	        bRect = Imgproc.boundingRect(squares.get(i));
	        
	        detectedLabel = src.submat(bRect);
	        
	        displayImage(detectedLabel, Integer.toString(i));
	        
       
	        try
	        {
	            //InputStream barCodeInputStream = new FileInputStream("c:\\users\\chris\\Documents\\detectedID.png");
	            //BufferedImage barCodeBufferedImage = ImageIO.read(barCodeInputStream);
	
		        source = new BufferedImageLuminanceSource((BufferedImage) toBufferedImage(detectedLabel));
		        bitmap = new BinaryBitmap(new HybridBinarizer(source));
		        reader = new MultiFormatReader();

		        barcodeResult = reader.decode(bitmap);
		        System.out.println(barcodeResult);

		        tessResult = tessInstance.doOCR((BufferedImage) toBufferedImage(detectedLabel));
		        System.out.println(tessResult);
        	}
            catch(Exception e)
            {
            	//System.out.println("Barcode not found.");
            	System.out.println("Barcode not found.");
	        }
	        
	        	source = null;
	        	bitmap = null;
	        	reader=null;
	        	barcodeResult = null;
	        	tessResult = null;
	        	
	        	i++;
	       	}
        // Save the visualized detection.
        //System.out.println("Writing "+ outFile);
        //Imgcodecs.imwrite(outFile, img);
    }
    
    public static void main(String[] args)
    {
		System.out.print(args[0] + ": ");
		
		System.gc();
		
		try{
			new BarcodePriceReader().run(args[0]);
			System.gc();
		}	
		catch(Exception e)
		{
		    e.printStackTrace();
		}
	}
}