package com.yeyupiaoling.changehumanbackground;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getName();
    private PaddleLiteSegmentation paddleLiteSegmentation;
    private ImageView imageView;
    private Bitmap resultPicture;
    private Bitmap humanPicture;
    private Bitmap changeBackgroundPicture;
    private Bitmap mergeBitmap1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        if (!hasPermission()) {
            requestPermission();
        }

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
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        String image_path;
        if (resultCode == Activity.RESULT_OK) {
            if (requestCode == 0) {
                if (data == null) {
                    Log.w("onActivityResult", "user photo data is null");
                    return;
                }
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
            } else if (requestCode == 1) {
                if (data == null) {
                    Log.w("onActivityResult", "user photo data is null");
                    return;
                }
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
            }
        }
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
        Utils.saveBitmap(frontBitmap);
        Bitmap bitmap = backBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(bitmap);
        Rect baseRect = new Rect(0, 0, backBitmap.getWidth(), backBitmap.getHeight());
        Rect frontRect = new Rect(0, 0, frontBitmap.getWidth(), frontBitmap.getHeight());
        canvas.drawBitmap(frontBitmap, frontRect, baseRect, null);
        return bitmap;
    }


    // check had permission
    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    // request permission
    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }
    }
}