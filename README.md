# android_live_yolo
android app using yolo

yolov5는 pytorch 기반으로 작성되어 있기 떄문에 weight가 pt(pytorch) 형식으로 작성되어 있다.
android는 학습된 모델을 tflite 확장자로 접근한다.
pt 확장자를 곧바로 tflite 확장자로 만들 수 없다.
pb(tensorflow) 확장자는 tflite와 같이 tensor형식으로 작성되어 있기에 변경이 쉽다.
따라서, pt -> onnx -> pb -> tflite 순으로 확장자를 변경해야 한다.
