package com.example.android_live_yolo;

import android.graphics.Rect;
import android.graphics.RectF;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;

import java.nio.ByteBuffer;
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
        List<ObjectPrediction> objectPredictionList = new ArrayList<>();

        tflite.runForMultipleInputsOutputs(new ByteBuffer[]{image.getBuffer()}, outputBuffer);

        for (int i=0;i<10;i++){
            ObjectPrediction objectPrediction = new ObjectPrediction();
            objectPrediction.location = new RectF(locations[i][0], locations[i][1], locations[i][2], locations[i][3]);
            objectPrediction.label = labels.get(1+Math.round(labelIndices[i]));
            objectPrediction.score = scores[i];
            objectPredictionList.add(objectPrediction);
        }

        return objectPredictionList;
    }

    public class ObjectPrediction{
        private RectF location;
        private String label;
        private float score;

        public float getScore() {
            return score;
        }
        public RectF getLocation() { return location; }
        public String getLabel() { return label; }
    }

}



