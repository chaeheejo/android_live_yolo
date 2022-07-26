package com.example.android_live_yolo;

import android.Manifest;
import android.graphics.Bitmap;
import android.os.Bundle;

import androidx.annotation.NonNull;
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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CameraFragment extends Fragment {

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private ExecutorService executor = Executors.newSingleThreadExecutor();
    private PreviewView previewView;
    private ImageButton camera_btn;
    private Bitmap bitmapBuffer;
    private int imageRotationDegrees;
    private Size tfInputSize;
    private Interpreter tflite;
    private Interpreter.Options options = new Interpreter.Options();
    private NnApiDelegate nnApiDelegate;
    private ImageProcessor processor;
    private TensorImage tfImageBuffer = new TensorImage(DataType.UINT8);
    private ObjectDetectionHelper detector;

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

        ActivityCompat.requestPermissions(getActivity(), new String[]{Manifest.permission.CAMERA}, 1);
        cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext());

        camera_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                tfload();
            }
        });

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
                analysisImage(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                Log.e("home", "at listener: ", e);
            }
        }, ContextCompat.getMainExecutor(requireContext()));

    }

    private void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder()
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        cameraProvider.bindToLifecycle((LifecycleOwner)this, cameraSelector, preview);
    }

    private void analysisImage(@NonNull ProcessCameraProvider cameraProvider){
        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetRotation(previewView.getDisplay().getRotation())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build();

        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {
                if (!bitmapBuffer.isRecycled()){
                    imageRotationDegrees = image.getImageInfo().getRotationDegrees();
                    bitmapBuffer = Bitmap.createBitmap(image.getWidth(), image.getHeight(),
                            Bitmap.Config.ARGB_8888);
                }
                if(pauseState){
                    image.close();
                    return;
                }

                bitmapBuffer.copyPixelsFromBuffer(image.getPlanes()[0].getBuffer());

                int imageSize = Math.min(bitmapBuffer.getHeight(), bitmapBuffer.getWidth());

                nnApiDelegate = new NnApiDelegate();
                options.addDelegate(nnApiDelegate);

                try {
                    tflite = new Interpreter(FileUtil.loadMappedFile(requireContext(), "coco_ssd_mobilenet_v1_1.0_quant.tflite"), options);
                }catch (IOException e){
                    Log.e("analyze", "tflite: "+e);
                }

                int[] shape = tflite.getInputTensor(0).shape();
                tfInputSize = new Size(shape[2], shape[1]);

                processor = new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(imageSize, imageSize))
                        .add(new ResizeOp(
                                tfInputSize.getHeight(), tfInputSize.getWidth(), ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new Rot90Op(-imageRotationDegrees / 90))
                        .add(new NormalizeOp(0f, 1f))
                        .build();

                tfImageBuffer.load(bitmapBuffer);
                TensorImage tfImage = processor.process(tfImageBuffer);

                try {
                    detector = new ObjectDetectionHelper(tflite, FileUtil.loadLabels(requireContext(), "coco_ssd_mobilenet_v1_1.0_labels.txt"));
                }catch (IOException e){
                    Log.e("analyze", "detector: "+e);
                }

            }
        });

    }

    private void tfload(){

    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        tflite.close();
        nnApiDelegate.close();
    }


}