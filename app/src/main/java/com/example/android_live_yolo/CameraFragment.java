package com.example.android_live_yolo;

import android.Manifest;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.LifecycleOwner;

import android.util.Log;
import android.util.Size;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;

import java.io.IOException;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CameraFragment extends Fragment {

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private ExecutorService executor;

    private PreviewView previewView;
    private ImageButton camera_btn;
    private TextView text_prediction;
    private View box_prediction;
    private ImageView image_predicted;

    private Bitmap bitmapBuffer;
    private int imageRotationDegrees;
    private Size tfInputSize;
    private boolean pauseState;

    private Interpreter tflite;
    private Interpreter.Options options;
    private NnApiDelegate nnApiDelegate;
    private ImageProcessor processor;
    private TensorImage tfImageBuffer;

    private ObjectDetectionHelper detector;
    private List<ObjectDetectionHelper.ObjectPrediction> prediction;

    public CameraFragment() {
    }
    public static CameraFragment newInstance(String param1, String param2) {
        CameraFragment fragment = new CameraFragment();
        Bundle args = new Bundle();
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        options = new Interpreter.Options();
        nnApiDelegate = new NnApiDelegate();
        tfImageBuffer = new TensorImage(DataType.UINT8);
        executor = Executors.newSingleThreadExecutor();
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_camera, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @NonNull Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        previewView = view.findViewById(R.id.camera_preview);
        camera_btn = view.findViewById(R.id.camera_capture_button);
        text_prediction = view.findViewById(R.id.text_prediction);
        box_prediction = view.findViewById(R.id.box_prediction);
        image_predicted = view.findViewById(R.id.image_predicted);

        ActivityCompat.requestPermissions(getActivity(), new String[]{Manifest.permission.CAMERA}, 1);
        cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext());

        camera_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                captureImage();
            }
        });

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                analysisImage(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                Log.e("home", "at listener: ", e);
            }
        }, ContextCompat.getMainExecutor(requireContext()));

    }

//    private void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
//        Preview preview = new Preview.Builder()
//                .build();
//
//        CameraSelector cameraSelector = new CameraSelector.Builder()
//                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
//                .build();
//
//        preview.setSurfaceProvider(previewView.getSurfaceProvider());
//
//        cameraProvider.unbindAll();
//        cameraProvider.bindToLifecycle((LifecycleOwner)this, cameraSelector, preview);
//    }

    private void analysisImage(@NonNull ProcessCameraProvider cameraProvider){
        Preview preview = new Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(previewView.getDisplay().getRotation())
                .build();

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(previewView.getDisplay().getRotation())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build();

        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {
                if (!bitmapBuffer.isRecycled()) {
                    imageRotationDegrees = image.getImageInfo().getRotationDegrees();
                    bitmapBuffer = Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);
                }
                if (pauseState) {
                    image.close();
                    return;
                }

                bitmapBuffer.copyPixelsFromBuffer(image.getPlanes()[0].getBuffer());
                int imageSize = Math.min(bitmapBuffer.getHeight(), bitmapBuffer.getWidth());

                options.addDelegate(nnApiDelegate);
                Log.d("camerafragment", "1success options");

                try {
                    tflite = new Interpreter(FileUtil.loadMappedFile(CameraFragment.this.requireContext(), "coco_ssd_mobilenet_v1_1.0_quant.tflite"), options);
                } catch (IOException e) {
                    Log.e("analyze", "tflite: " + e);
                }

                Log.d("camerafragment", "2success tflite");

                int[] shape = tflite.getInputTensor(0).shape();
                tfInputSize = new Size(shape[2], shape[1]);

                processor = new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(imageSize, imageSize))
                        .add(new ResizeOp(tfInputSize.getHeight(), tfInputSize.getWidth(), ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new Rot90Op(-imageRotationDegrees / 90))
                        .add(new NormalizeOp(0f, 1f))
                        .build();

                tfImageBuffer.load(bitmapBuffer);
                TensorImage tfImage = processor.process(tfImageBuffer);

                Log.d("camerafragment", "3success processor");

                try {
                    detector = new ObjectDetectionHelper(tflite, FileUtil.loadLabels(CameraFragment.this.requireContext(), "coco_ssd_mobilenet_v1_1.0_labels.txt"));
                } catch (IOException e) {
                    Log.e("analyze", "detector: " + e);
                }

                prediction = detector.predict(tfImage);
                Log.d("camerafragment", "predict: " + prediction);

                Collections.sort(prediction, new Comparator<ObjectDetectionHelper.ObjectPrediction>() {
                    @Override
                    public int compare(ObjectDetectionHelper.ObjectPrediction o1, ObjectDetectionHelper.ObjectPrediction o2) {
                        return Math.round(o1.getScore()) - Math.round(o2.getScore());
                    }
                });

                CameraFragment.this.drawBox(prediction.get(0));
            }
        });

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        cameraProvider.unbindAll();
        cameraProvider.bindToLifecycle((LifecycleOwner)this, cameraSelector, preview);
    }

    private void drawBox(ObjectDetectionHelper.ObjectPrediction predict){
        if (predict==null || predict.getScore()<0.5){
            box_prediction.setVisibility(View.GONE);
            text_prediction.setVisibility(View.GONE);
            return;
        }

        RectF location = mapLocation(predict.getLocation());

        text_prediction.setText(String.format("%.2f", predict.getScore())+predict.getLabel());
        ViewGroup.MarginLayoutParams lp = (ViewGroup.MarginLayoutParams) box_prediction.getLayoutParams();
        lp.setMargins(
                Math.round(location.top),
                Math.round(location.left),
                Math.min(previewView.getWidth(), Math.round(location.right - location.left)),
                Math.min(previewView.getHeight(), Math.round(location.bottom - location.top))
        );
        box_prediction.setLayoutParams(lp);

        text_prediction.setVisibility(View.VISIBLE);
        box_prediction.setVisibility(View.VISIBLE);
    }

    private RectF mapLocation(RectF location){
        RectF previewLocation = new RectF(
                location.left * previewView.getWidth(),
                location.top * previewView.getHeight(),
                location.right * previewView.getWidth(),
                location.bottom * previewView.getHeight()
        );

        int lensFacing = CameraSelector.LENS_FACING_BACK;
        boolean isFrontFacing = lensFacing == CameraSelector.LENS_FACING_FRONT;
        RectF mirrorLocation;
        if (isFrontFacing){
             mirrorLocation = new RectF(
                    previewView.getWidth() - previewLocation.right,
                    previewLocation.top,
                    previewView.getWidth() - previewLocation.left,
                    previewLocation.bottom
            );
        }
        else{
            mirrorLocation = new RectF(previewLocation);
        }

        float midX = (mirrorLocation.left + mirrorLocation.right) / 2f;
        float midY = (mirrorLocation.top + mirrorLocation.bottom) / 2f;
        RectF marginLocation;
        if (previewView.getWidth() < previewView.getHeight()){
            marginLocation = new RectF(
                    midX - (1f + 0.1f) * (4f / 3f) * mirrorLocation.width() / 2f,
                    midY - (1f - 0.1f) * mirrorLocation.height() / 2f,
                    midX + (1f + 0.1f) * (4f / 3f) * mirrorLocation.width() / 2f,
                    midY + (1f - 0.1f) * mirrorLocation.height() / 2f
            );
        } else {
            marginLocation = new RectF(
                    midX - (1f - 0.1f) * mirrorLocation.width() / 2f,
                    midY - (1f + 0.1f) * (4f / 3f) * mirrorLocation.height() / 2f,
                    midX + (1f - 0.1f) * mirrorLocation.width() / 2f,
                    midY + (1f + 0.1f) * (4f / 3f) * mirrorLocation.height() / 2f
            );
        }

        return marginLocation;
    }

    private void captureImage(){

    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        tflite.close();
        nnApiDelegate.close();
    }


}