package com.example.gesture;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.app.Application;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.StringWriter;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import org.tensorflow.Graph;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.lite.Interpreter;

public class MainActivity extends AppCompatActivity{

    ImageView imageView;

    protected Interpreter tflite;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("OpenCV", "OpenCV loaded successfully");
//                    imageMat=new Mat();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        OpenCVLoader.initDebug();

        imageView = findViewById(R.id.imageView);

    }

    public void displayToast(View v) throws IOException {

        Bitmap bmp = BitmapFactory.decodeResource(getResources(),R.drawable.c);

        Mat tmp = new Mat(bmp.getWidth(), bmp.getHeight(), CvType.CV_8U);

        Utils.bitmapToMat(bmp, tmp);

        Imgproc.cvtColor(tmp, tmp, Imgproc.COLOR_RGB2GRAY);

        AssetFileDescriptor fileDescriptor = getAssets().openFd("tf_model.pb");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();

        Imgproc.threshold(tmp, tmp, 40, 255, Imgproc.THRESH_BINARY);

        tmp = pointremove(tmp);

        Imgproc.GaussianBlur(tmp, tmp, new Size(5, 5), 0 ,  0);

        Imgproc.threshold(tmp,tmp,130,255,Imgproc.THRESH_OTSU);

        Mat image = tmp.clone();
        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroid = new Mat();

        int a = Imgproc.connectedComponentsWithStats(image,labels,stats,centroid);

        Log.d("Array"," ");

        double area = stats.get(1,4)[0];
        int max_label = 1;
        for(int i=1;i<a;i++){
            if (stats.get(i,4)[0] > area){
                area = stats.get(i,4)[0];
                max_label = i;
                Log.d("hello","hello");
            }
        }

        double [] left = stats.get(max_label,0);
        double [] top = stats.get(max_label,1);
        double [] width = stats.get(max_label,2);
        double [] height = stats.get(max_label,3);

        Mat C = new Mat((int) height[0], (int) width[0], CvType.CV_8U);

        Log.d("Arraydo"," " + left[0] + " " + top[0] + " " + width[0] + " " + height[0]);

        for(int i = (int) left[0]; i< left[0] + width[0]; i++){
            for(int j = (int) top[0]; j<top[0] + height[0]; j++){
                double [] apk1 = new double[1];
                apk1[0] = 0.0;
                C.put((int) (j-top[0]), (int) (i-left[0]),apk1);
                if(labels.get(j,i)[0] == max_label){
                    double [] apk = new double[1];
                    apk[0] = 255.0;
                    C.put((j - (int) top[0]),(i - (int) left[0]),apk);
                }
            }
        }

        Bitmap Cbitmap = Bitmap.createScaledBitmap(bmp,(int)width[0],(int)height[0],false);

        Log.d("Array"," " + left.length + " " + top.length + " " + width.length + " " + height.length);

        Log.d("ArrayC",C.height() + " " + C.width());

        Mat final_image = new Mat(100, 100, CvType.CV_8U);

        Size sz = new Size(100,100);

        Imgproc.resize(C,final_image,(sz));

        Log.d("ArrayCbit","" + C.height() + " " + Cbitmap.getHeight() + " " + C.width() + " " + Cbitmap.getWidth());

        Bitmap final_bitmap = Bitmap.createScaledBitmap(bmp,100,100,false);

        String output = "";

        Utils.matToBitmap(final_image, final_bitmap);

        tflite.run(final_bitmap, output);

        imageView.setImageBitmap(final_bitmap);
    }

    private Mat pointremove(Mat tmp) {

        Mat image = tmp.clone();
        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroid = new Mat();

        int a = Imgproc.connectedComponentsWithStats(image,labels,stats,centroid);

        Log.d("Array"," ");

        for(int i=0;i<a;i++){
                if (stats.get(i,4)[0] < 100){
                    double [] left = stats.get(i,0);
                    double [] top = stats.get(i,1);
                    double [] width = stats.get(i,2);
                    double [] height = stats.get(i,3);

                    Log.d("Arraydo"," " + left[0] + " " + top[0] + " " + width[0] + " " + height[0]);

                    for(int k = (int) left[0]; k< left[0] + width[0]; k++){
                        for(int l = (int) top[0]; l<top[0] + height[0]; l++){
                            double [] apk1 = new double[1];
                            apk1[0] = 0.0;
                            tmp.put(l, k, apk1);
                        }
                    }
                }

        }
        return tmp;
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

    }

}
