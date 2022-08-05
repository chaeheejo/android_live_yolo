# android_live_yolo

### Description
> 실시간 물체 감지를 하는 앱

<br/>

### development logic
+ android CameraX를 활용해 화면에 카메라 preview를 띄워준다.
+ 얻은 preview에 대해 tensorflow lite 모델을 적용해 object detection을 진행한다.
+ 결과 값을 카메라 preview에 다시 띄워준다.
+ 버튼을 누르면 내부 저장소에 사진이 저장된다.
