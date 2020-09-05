# ChangeHumanBackground
人物更换背景

> 原文博客：[Doi技术团队](http://blog.doiduoyi.com)
> 链接地址：[https://blog.doiduoyi.com/authors/1584446358138](https://blog.doiduoyi.com/authors/1584446358138)
> 初心：记录优秀的Doi技术团队学习经历
>本文链接：[Android基于图像语义分割实现人物背景更换](https://blog.doiduoyi.com/articles/1598712095900.html)


本教程是通过PaddlePaddle的PaddleSeg实现的，该开源库的地址为：[http://github.com/PaddlPaddle/PaddleSeg](http://github.com/PaddlPaddle/PaddleSeg) ，使用开源库提供的预训练模型实现人物的图像语义分割，最终部署到Android应用上。关于如何在Android应用上使用PaddlePaddle模型，可以参考笔者的这篇文章[《基于Paddle Lite在Android手机上实现图像分类》](https://blog.doiduoyi.com/articles/1596345808188.html)。

**本教程开源代码地址：[https://github.com/yeyupiaoling/ChangeHumanBackground](https://github.com/yeyupiaoling/ChangeHumanBackground)**

# 图像语义分割工具
首先编写一个可以在Android应用使用PaddlePaddle的图像语义分割模型的工具类，通过是这个`PaddleLiteSegmentation`这个java工具类实现模型的加载和图像的预测。

首先是加载模型，获得一个预测器，其中`inputShape`为图像的输入大小，`NUM_THREADS`为使用线程数来预测图像，最高可以支持4个线程预测。
```java
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
```

在预测开始之前，写两个重构方法，这个我们这个工具不管是图片路径还是图像的Bitmap都可以实现语义分割了。
```java
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
```

现在还不能预测，还需要对图像进行预处理的方法，预测器输入的是一个浮点数组，而不是一个Bitmap对象，所以需要这样的一个工具方法，把图像Bitmap转换为浮点数组，同时对图像进行预处理，如通道顺序的变换，有的模型还需要数据的标准化，但这里没有使用到。
```java
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
```

最后就可以执行预测了，预测的结果是一个数组，它代表了整个图像的语义分割的情况，0的为背景，1的为人物。
```java
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
```


# 实现人物背景更换

在`MainActivity`中，程序加载的时候就从assets中把模型复制到缓存目录中，然后加载图像语义分割模型。

```java
String segmentationModelPath = getCacheDir().getAbsolutePath() + File.separator + "model.nb";
Utils.copyFileFromAsset(MainActivity.this, "model.nb", segmentationModelPath);
try {
    paddleLiteSegmentation = new PaddleLiteSegmentation(segmentationModelPath);
    Toast.makeText(MainActivity.this, "模型加载成功！", Toast.LENGTH_SHORT).show();
    Log.d(TAG, "模型加载成功！");
} catch (Exception e) {
    Toast.makeText(MainActivity.this, "模型加载失败！", Toast.LENGTH_SHORT).show();
    Log.d(TAG, "模型加载失败！");
    e.printStackTrace();
    finish();
}
```

创建几个按钮，来控制图片背景的更换。
```java
// 获取控件
Button selectPicture = findViewById(R.id.select_picture);
Button selectBackground = findViewById(R.id.select_background);
Button savePicture = findViewById(R.id.save_picture);
imageView = findViewById(R.id.imageView);
selectPicture.setOnClickListener(new View.OnClickListener() {
    @Override
    public void onClick(View v) {
        // 打开相册
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, 0);
    }
});
selectBackground.setOnClickListener(new View.OnClickListener() {
    @Override
    public void onClick(View v) {
        if (resultPicture != null){
            // 打开相册
            Intent intent = new Intent(Intent.ACTION_PICK);
            intent.setType("image/*");
            startActivityForResult(intent, 1);
        }else {
            Toast.makeText(MainActivity.this, "先选择人物图片！", Toast.LENGTH_SHORT).show();
        }
    }
});
savePicture.setOnClickListener(new View.OnClickListener() {
    @Override
    public void onClick(View v) {
        // 保持图片
        String savePth = Utils.saveBitmap(mergeBitmap1);
        if (savePth != null) {
            Toast.makeText(MainActivity.this, "图片保存：" + savePth, Toast.LENGTH_SHORT).show();
            Log.d(TAG, "图片保存：" + savePth);
        } else {
            Toast.makeText(MainActivity.this, "图片保存失败", Toast.LENGTH_SHORT).show();
            Log.d(TAG, "图片保存失败");
        }
    }
});
```

首先需要选择包含人物的图片，这时就需要对图像进行预测，获取语义分割结果，然后将图像放大的跟原图像一样大小，并做这个临时的画布。
```java
Uri image_uri = data.getData();
image_path = Utils.getPathFromURI(MainActivity.this, image_uri);
try {
    // 预测图像
    FileInputStream fis = new FileInputStream(image_path);
    Bitmap b = BitmapFactory.decodeStream(fis);
    long start = System.currentTimeMillis();
    long[] result = paddleLiteSegmentation.predictImage(image_path);
    long end = System.currentTimeMillis();

    // 创建一个任务为全黑色，背景完全透明的图片
    humanPicture = b.copy(Bitmap.Config.ARGB_8888, true);
    final int[] colors_map = {0x00000000, 0xFF000000};
    int[] objectColor = new int[result.length];

    for (int i = 0; i < result.length; i++) {
        objectColor[i] = colors_map[(int) result[i]];
    }
    Bitmap.Config config = humanPicture.getConfig();
    Bitmap outputImage = Bitmap.createBitmap(objectColor, (int) PaddleLiteSegmentation.inputShape[2], (int) PaddleLiteSegmentation.inputShape[3], config);
    resultPicture = Bitmap.createScaledBitmap(outputImage, humanPicture.getWidth(), humanPicture.getHeight(), true);

    imageView.setImageBitmap(b);
    Log.d(TAG, "预测时间：" + (end - start) + "ms");
} catch (Exception e) {
    e.printStackTrace();
}
```

最后在这里实现人物背景的更换，
```java
Uri image_uri = data.getData();
image_path = Utils.getPathFromURI(MainActivity.this, image_uri);
try {
    FileInputStream fis = new FileInputStream(image_path);
    changeBackgroundPicture = BitmapFactory.decodeStream(fis);
    mergeBitmap1 = draw();
    imageView.setImageBitmap(mergeBitmap1);
} catch (Exception e) {
    e.printStackTrace();
}

// 实现换背景
public Bitmap draw() {
    // 创建一个对应人物位置透明其他正常的背景图
    Bitmap bgBitmap = Bitmap.createScaledBitmap(changeBackgroundPicture, resultPicture.getWidth(), resultPicture.getHeight(), true);
    for (int y = 0; y < resultPicture.getHeight(); y++) {
        for (int x = 0; x < resultPicture.getWidth(); x++) {
            int color = resultPicture.getPixel(x, y);
            int a = Color.alpha(color);
            if (a == 255) {
                bgBitmap.setPixel(x, y, Color.TRANSPARENT);
            }
        }
    }

    // 添加画布，保证透明
    Bitmap bgBitmap2 = Bitmap.createBitmap(bgBitmap.getWidth(), bgBitmap.getHeight(), Bitmap.Config.ARGB_8888);
    Canvas canvas1 = new Canvas(bgBitmap2);
    canvas1.drawBitmap(bgBitmap, 0, 0, null);

    return mergeBitmap(humanPicture, bgBitmap2);
}

// 合并两张图片
public static Bitmap mergeBitmap(Bitmap backBitmap, Bitmap frontBitmap) {
    Bitmap bitmap = backBitmap.copy(Bitmap.Config.ARGB_8888, true);
    Canvas canvas = new Canvas(bitmap);
    Rect baseRect = new Rect(0, 0, backBitmap.getWidth(), backBitmap.getHeight());
    Rect frontRect = new Rect(0, 0, frontBitmap.getWidth(), frontBitmap.getHeight());
    canvas.drawBitmap(frontBitmap, frontRect, baseRect, null);
    return bitmap;
}
```

实现的效果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200829223431600.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70#pic_center)
