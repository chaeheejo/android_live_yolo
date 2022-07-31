package com.example.android_live_yolo

import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage

class ObjectDetection(private val tflite: Interpreter, private val labels: List<String>) {

    data class ObjectPrediction(val location: RectF, val label: String, val score: Float)

    private val locations = arrayOf(Array(OBJECT_COUNT) { FloatArray(4) })
    private val labelIndices =  arrayOf(FloatArray(OBJECT_COUNT))
    private val scores =  arrayOf(FloatArray(OBJECT_COUNT))

    private val outputBuffer = mapOf(
        0 to locations,
        1 to labelIndices,
        2 to scores,
        3 to FloatArray(1)
    )

    val predictions get() = (0 until OBJECT_COUNT).map {
        ObjectPrediction(

            location = locations[0][it].let {
                RectF(it[1], it[0], it[3], it[2])
            },

            label = labels[1 + labelIndices[0][it].toInt()],

            score = scores[0][it]
        )
    }

    fun predict(image: TensorImage): List<ObjectPrediction> {
        tflite.runForMultipleInputsOutputs(arrayOf(image.buffer), outputBuffer)
        return predictions
    }

    companion object {
        const val OBJECT_COUNT = 10
    }
}