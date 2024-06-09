package com.example.myapp;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Color;
import android.os.Bundle;

import androidx.annotation.Nullable;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;


import com.example.myapp.ml.FlowerClassificationOptimized;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;


public class MainActivity extends AppCompatActivity {

    Button camera, gallery;
    ImageView imageView;
    TextView result;
    int imageSize = 256;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
    }

    public void classifyImage(Bitmap image) {
        try {
            // Resize the image to match the input size of the model (256x256)
            Bitmap resizedImage = Bitmap.createScaledBitmap(image, 256, 256, true);

            // Normalize pixel values to the range [0, 1]
            float[] normalizedPixels = new float[256 * 256 * 3];
            int pixelIndex = 0;
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    int pixel = resizedImage.getPixel(j, i);
                    // Extract RGB components and normalize
                    normalizedPixels[pixelIndex++] = Color.red(pixel) / 255.0f;
                    normalizedPixels[pixelIndex++] = Color.green(pixel) / 255.0f;
                    normalizedPixels[pixelIndex++] = Color.blue(pixel) / 255.0f;
                }
            }

            // Prepare input buffer for the model
            TensorBuffer inputBuffer = TensorBuffer.createFixedSize(new int[]{1, 256, 256, 3}, DataType.FLOAT32);
            inputBuffer.loadArray(normalizedPixels, new int[]{1, 256, 256, 3});

            // Run inference with the model
            FlowerClassificationOptimized model = FlowerClassificationOptimized.newInstance(getApplicationContext());
            FlowerClassificationOptimized.Outputs outputs = model.process(inputBuffer);
            TensorBuffer outputBuffer = outputs.getOutputFeature0AsTensorBuffer();

            // Process output buffer to get predicted class
            float[] outputScores = outputBuffer.getFloatArray();
            int maxPos = 0;
            float maxConfidence = outputScores[0];
            for (int i = 1; i < outputScores.length; i++) {
                if (outputScores[i] > maxConfidence) {
                    maxConfidence = outputScores[i];
                    maxPos = i;
                }
            }

            // Map class index to class label
            String[] classes = {"black_eyed_susan", "calendula", "california_poppy", "common_daisy", "coreopsis", "dandelion", "iris", "rose", "sunflower", "tulip"};
            String predictedClass = classes[maxPos];
            result.setText(predictedClass);

            // Release model resources
            model.close();
        } catch (IOException e) {
            // Handle IOException
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode == RESULT_OK){
            if(requestCode == 3){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }else{
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}