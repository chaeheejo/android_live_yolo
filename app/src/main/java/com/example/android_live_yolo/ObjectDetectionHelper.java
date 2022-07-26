package com.example.android_live_yolo;

import android.graphics.RectF;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;

import java.sql.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class ObjectDetectionHelper {

    private float[][] locations = new float[10][4];
    private float[] labelIndices = new float[10];
    private float[] scores = new float[10];
    private Interpreter tflite;
    private List<String> labels;
    private Map outputBuffer;

    public ObjectDetectionHelper(Interpreter tflite, List<String> labels) {
        this.tflite = tflite;
        this.labels = labels;

        float[] tmp = new float[1];

        outputBuffer = Map.of(
                0, locations,
                1, labelIndices,
                2, scores,
                3, tmp
        );
    }

    public List<ObjectPrediction> predict(TensorImage image){
        List<ObjectPrediction> objectPrediction = new ArrayList<>();

        tflite.runForMultipleInputsOutputs(image.getBuffer(), outputBuffer);

        return objectPrediction;
    }

    public class ObjectPrediction{
        private RectF location;
        private String label;
        private Float score;
    }


}



