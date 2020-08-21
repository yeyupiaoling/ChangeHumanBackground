package com.yeyupiaoling.changehumanbackground;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.util.Log;

import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.PowerMode;
import com.baidu.paddle.lite.Tensor;

import java.io.File;
import java.io.FileInputStream;
import java.util.Arrays;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;

public class PaddleLiteSegmentation {
    private static final String TAG = PaddleLiteSegmentation.class.getName();

    private PaddlePredictor paddlePredictor;
    private Tensor inputTensor;
    public static long[] inputShape = new long[]{1, 3, 513, 513};
    private static final int NUM_THREADS = 4;

    /**
     * @param modelPath model path
     */
    public PaddleLiteSegmentation(String modelPath) throws Exception {
        File file = new File(modelPath);
        if (!file.exists()) {
            throw new Exception("model file is not exists!");
        }
        try {
            MobileConfig config = new MobileConfig();
            config.setModelFromFile(modelPath);
            config.setThreads(NUM_THREADS);
            config.setPowerMode(PowerMode.LITE_POWER_HIGH);
            paddlePredictor = PaddlePredictor.createPaddlePredictor(config);

            inputTensor = paddlePredictor.getInput(0);
            inputTensor.resize(inputShape);
        } catch (Exception e) {
            e.printStackTrace();
            throw new Exception("load model fail!");
        }
    }

    public long[] predictImage(String image_path) throws Exception {
        if (!new File(image_path).exists()) {
            throw new Exception("image file is not exists!");
        }
        FileInputStream fis = new FileInputStream(image_path);
        Bitmap bitmap = BitmapFactory.decodeStream(fis);
        long[] result = predictImage(bitmap);
        if (bitmap.isRecycled()) {
            bitmap.recycle();
        }
        return result;
    }

    public long[] predictImage(Bitmap bitmap) throws Exception {
        return predict(bitmap);
    }


    // 预测
    private long[] predict(Bitmap bmp) throws Exception {
        float[] inputData = getScaledMatrix(bmp);
        inputTensor.setData(inputData);

        try {
            paddlePredictor.run();
        } catch (Exception e) {
            throw new Exception("predict image fail! log:" + e);
        }
        Tensor outputTensor = paddlePredictor.getOutput(0);
        long[] output = outputTensor.getLongData();
        long[] outputShape = outputTensor.shape();
        Log.d(TAG, "结果shape："+ Arrays.toString(outputShape));
        return output;
    }


    // 对将要预测的图片进行预处理
    private float[] getScaledMatrix(Bitmap bitmap) {
        int channels = (int) inputShape[1];
        int width = (int) inputShape[2];
        int height = (int) inputShape[3];
        float[] inputData = new float[channels * width * height];
        Bitmap rgbaImage = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Bitmap scaleImage = Bitmap.createScaledBitmap(rgbaImage, width, height, true);
        Log.d(TAG, scaleImage.getWidth() +  ", " + scaleImage.getHeight());

        if (channels == 3) {
            // RGB = {0, 1, 2}, BGR = {2, 1, 0}
            int[] channelIdx = new int[]{0, 1, 2};
            int[] channelStride = new int[]{width * height, width * height * 2};
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int color = scaleImage.getPixel(x, y);
                    float[] rgb = new float[]{(float) red(color), (float) green(color), (float) blue(color)};
                    inputData[y * width + x] = rgb[channelIdx[0]];
                    inputData[y * width + x + channelStride[0]] = rgb[channelIdx[1]];
                    inputData[y * width + x + channelStride[1]] = rgb[channelIdx[2]];
                }
            }
        } else if (channels == 1) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int color = scaleImage.getPixel(x, y);
                    float gray = (float) (red(color) + green(color) + blue(color));
                    inputData[y * width + x] = gray;
                }
            }
        } else {
            Log.e(TAG, "图片的通道数必须是1或者3");
        }
        return inputData;
    }
}
