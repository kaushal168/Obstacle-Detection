/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.ar.core.examples.java.ml

import android.opengl.Matrix
import android.util.Log
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.*
import com.google.ar.core.examples.java.common.helpers.DisplayRotationHelper
import com.google.ar.core.examples.java.common.samplerender.SampleRender
import com.google.ar.core.examples.java.common.samplerender.arcore.BackgroundRenderer
import com.google.ar.core.examples.java.ml.render.LabelRender
import com.google.ar.core.examples.java.ml.render.PointCloudRender
import com.google.ar.core.examples.java.ml.utils.TTS_Conversion
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.sceneform.ux.ArFragment
import java.util.Collections
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import java.util.ArrayList
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Renders the HelloAR application into using our example Renderer.
 */
class AppRenderer(val activity: MainActivity) : DefaultLifecycleObserver, SampleRender.Renderer, CoroutineScope by MainScope() {
  companion object {
    val TAG = "HelloArRenderer"
  }
  val map_labels= HashMap<Pair<String, String>, Int>()

  lateinit var view: MainActivityView

  val displayRotationHelper = DisplayRotationHelper(activity)
  lateinit var backgroundRenderer: BackgroundRenderer
  val pointCloudRender = PointCloudRender()
  val labelRenderer = LabelRender()

    val viewMatrix = FloatArray(16)
    val projectionMatrix = FloatArray(16)
    val viewProjectionMatrix = FloatArray(16)

  val arLabeledAnchors = Collections.synchronizedList(mutableListOf<ARLabeledAnchor>())
  var scanButtonWasPressed = false

  val mlKitAnalyzer = MLKitObjectDetector(activity)
  val gcpAnalyzer = GoogleCloudVisionDetector(activity)

  var currentAnalyzer: ObjectDetector = gcpAnalyzer

  private var arFragment: ArFragment? = null
  private val placedAnchors = ArrayList<Anchor>()

  private fun changeUnit(distanceMeter: Float, unit: String): Float{
    // Converts distance units.
    return when(unit){
      "cm" -> distanceMeter * 100
      "mm" -> distanceMeter * 1000
      else -> distanceMeter
    }
  }

  private fun makeDistanceTextWithCM(distanceMeter: Float): String{
    // Formats distance with centimeters.
    val distanceCM = changeUnit(distanceMeter, "cm")
    val distanceCMFloor = "%.2f".format(distanceCM)
    return "${distanceCMFloor} cm"
  }

  private fun measureDistanceOf2Points(distanceMeter: Float): String{
    // Measures distance between two points.
    val distanceTextCM = makeDistanceTextWithCM(distanceMeter)
    Log.d("measurement", "distance of 2 Points: ${distanceTextCM}")
    return distanceTextCM
  }

  private fun measureDistanceFromCamera(objectPose: Pose, cameraPose: Pose): String {
    // Measures distance from object to camera.
    val distanceMeter = calculateDistance(objectPose, cameraPose)
    val distanceTextCM = makeDistanceTextWithCM(distanceMeter)
    Log.d("measurement", "Distance from camera: $distanceTextCM")
    if(distanceMeter < 1) {
      showSnackbar("Distance is less than 100 cm")
    }
    return distanceTextCM
  }


  private fun calculateDistance(x: Float, y: Float, z:Float): Float{
    // Calculates Euclidean distance.
    return sqrt(x.pow(2) + y.pow(2) + z.pow(2))
  }

  private fun calculateDistance(objectPose0: Pose, objectPose1: Pose): Float{
    // Calculates distance between two poses.
    return calculateDistance(
      objectPose0.tx() - objectPose1.tx(),
      objectPose0.ty() - objectPose1.ty(),
      objectPose0.tz() - objectPose1.tz()
    )
  }

  override fun onResume(owner: LifecycleOwner) {
    displayRotationHelper.onResume()
  }

  override fun onPause(owner: LifecycleOwner) {
    displayRotationHelper.onPause()
  }

  fun bindView(view: MainActivityView) {
    this.view = view

    view.scanButton.setOnClickListener {
      // frame.acquireCameraImage is dependent on an ARCore Frame, which is only available in onDrawFrame.
      // Use a boolean and check its state in onDrawFrame to interact with the camera image.
      scanButtonWasPressed = true
      view.setScanningActive(true)
      hideSnackbar()
    }

    view.useCloudMlSwitch.setOnCheckedChangeListener { _, isChecked ->
      currentAnalyzer = if (isChecked) gcpAnalyzer else mlKitAnalyzer
    }

    val gcpConfigured = gcpAnalyzer.credentials != null
    view.useCloudMlSwitch.isChecked = gcpConfigured
    view.useCloudMlSwitch.isEnabled = gcpConfigured
    currentAnalyzer = if (gcpConfigured) gcpAnalyzer else mlKitAnalyzer

    if (!gcpConfigured) {
      showSnackbar("Google Cloud Vision isn't configured (see README). The Cloud ML switch will be disabled.")
    }

    view.resetButton.setOnClickListener {
      arLabeledAnchors.clear()
      view.resetButton.isEnabled = false
      hideSnackbar()
    }
  }

  override fun onSurfaceCreated(render: SampleRender) {
    backgroundRenderer = BackgroundRenderer(render).apply {
      setUseDepthVisualization(render, false)
    }
    pointCloudRender.onSurfaceCreated(render)
    labelRenderer.onSurfaceCreated(render)
  }

  override fun onSurfaceChanged(render: SampleRender?, width: Int, height: Int) {
    displayRotationHelper.onSurfaceChanged(width, height)
  }

  var objectResults: List<DetectedObjectResult>? = null

  override fun onDrawFrame(render: SampleRender) {
    val session = activity.arCoreSessionHelper.sessionCache ?: return
    session.setCameraTextureNames(intArrayOf(backgroundRenderer.cameraColorTexture.textureId))

    // Notify ARCore session that the view size changed so that the perspective matrix and
    // the video background can be properly adjusted.
    displayRotationHelper.updateSessionIfNeeded(session)

    val frame = try {
      session.update()
    } catch (e: CameraNotAvailableException) {
      // Handle camera not available exception
      Log.e(TAG, "Camera not available during onDrawFrame", e)
      showSnackbar("Camera not available. Try restarting the app.")
      return
    }

    backgroundRenderer.updateDisplayGeometry(frame)
    backgroundRenderer.drawBackground(render)

    // Get camera and projection matrices.
    val camera = frame.camera
    camera.getViewMatrix(viewMatrix, 0)
    camera.getProjectionMatrix(projectionMatrix, 0, 0.01f, 100.0f)
    Matrix.multiplyMM(viewProjectionMatrix, 0, projectionMatrix, 0, viewMatrix, 0)

    // Handle tracking failures.
    if (camera.trackingState != TrackingState.TRACKING) {
      return
    }

    // Draw point cloud.
    frame.acquirePointCloud().use { pointCloud ->
      pointCloudRender.drawPointCloud(render, pointCloud, viewProjectionMatrix)
    }

    // Frame.acquireCameraImage must be used on the GL thread.
    // Check if the button was pressed last frame to start processing the camera image.
    if (scanButtonWasPressed) {
      scanButtonWasPressed = false
      val cameraImage = frame.tryAcquireCameraImage()
      if (cameraImage != null) {
        // Call our ML model on an IO thread.
        launch(Dispatchers.IO) {
          val cameraId = session.cameraConfig.cameraId
          val imageRotation = displayRotationHelper.getCameraSensorToDisplayRotation(cameraId)
          objectResults = currentAnalyzer.analyze(cameraImage, imageRotation)


//          for(i in objectResults!!){
//            var tts = TTS_Conversion(view.root.context,i.label+" "+(i.confidence*100).toInt()+" percent accuracy")
//          }

          cameraImage.close()
        }
      }
    }

    /** If results were completed this frame, create [Anchor]s from model results. */
    val objects = objectResults
    val cameraPose = frame.camera.pose
    if (objects != null) {
      objectResults = null
      Log.i(TAG, "$currentAnalyzer got objects: $objects")
      val anchors = objects.mapNotNull { obj ->
        val (atX, atY) = obj.centerCoordinate
        val anchor = createAnchor(atX.toFloat(), atY.toFloat(), frame) ?: return@mapNotNull null
        Log.i(TAG, "Created anchor ${anchor.pose} from hit test")
        placedAnchors.add(anchor)
        val distance = measureDistanceFromCamera(anchor.pose, cameraPose) // Pass cameraPose instead of frame
        ARLabeledAnchor(anchor, obj.label+" "+distance)
      }
      arLabeledAnchors.addAll(anchors)
      view.post {
        view.resetButton.isEnabled = arLabeledAnchors.isNotEmpty()
        view.setScanningActive(false)
        when {
          objects.isEmpty() && currentAnalyzer == mlKitAnalyzer && !mlKitAnalyzer.hasCustomModel() ->
            showSnackbar("Default ML Kit classification model returned no results. " +
              "For better classification performance, see the README to configure a custom model.")
          objects.isEmpty() ->
            showSnackbar("Classification model returned no results.")
          anchors.size != objects.size ->
            showSnackbar("Objects were classified, but could not be attached to an anchor. " +
              "Try moving your device around to obtain a better understanding of the environment.")
        }
      }
    }

    // Draw labels at their anchor position.
    for (arDetectedObject in arLabeledAnchors) {
      val anchor = arDetectedObject.anchor
      if (anchor.trackingState != TrackingState.TRACKING) continue

      labelRenderer.draw(
        render,
        viewProjectionMatrix,
        anchor.pose,
        camera.pose,
        arDetectedObject.label
      )
      val arr_str=arDetectedObject.label.split(" ");
      val j=0
      var arr_str_size1 = arr_str.size-3
      var object_name = "";
      for (j in 0..arr_str_size1){
        object_name += arr_str[j];
      }
      var object_distance = "";
      for ( j in arr_str_size1+1..arr_str.size-1){
        object_distance += arr_str[j];
      }
      if(!map_labels.containsKey(Pair(object_name,object_distance))){

        val tell=object_name + " is found at the position "+object_distance;
        var tts = TTS_Conversion(view.root.context,tell);
        map_labels.put(Pair(object_name,object_distance),1);
      }

      val distanceText = measureDistanceFromCamera(anchor.pose, cameraPose) // Pass the camera pose here
      val originalLabel = arDetectedObject.label
      val newLabel = originalLabel.replace("\\d+(\\.\\d+)?\\s*cm".toRegex(), "").trim()
      val finalLabel = "$newLabel - $distanceText"
      labelRenderer.draw(render, viewProjectionMatrix, anchor.pose, cameraPose, finalLabel)

    }
  }

  /**
   * Utility method for [Frame.acquireCameraImage] that maps [NotYetAvailableException] to `null`.
   */
  fun Frame.tryAcquireCameraImage() = try {
    acquireCameraImage()
  } catch (e: NotYetAvailableException) {
    null
  } catch (e: Throwable) {
    throw e
  }

  private fun showSnackbar(message: String): Unit =
    activity.view.snackbarHelper.showError(activity, message)

  private fun hideSnackbar() = activity.view.snackbarHelper.hide(activity)

  /**
   * Temporary arrays to prevent allocations in [createAnchor].
   */
  private val convertFloats = FloatArray(4)
  private val convertFloatsOut = FloatArray(4)

  /** Create an anchor using (x, y) coordinates in the [Coordinates2d.IMAGE_PIXELS] coordinate space. */
  fun createAnchor(xImage: Float, yImage: Float, frame: Frame): Anchor? {
    // IMAGE_PIXELS -> VIEW
    convertFloats[0] = xImage
    convertFloats[1] = yImage
    frame.transformCoordinates2d(
      Coordinates2d.IMAGE_PIXELS,
      convertFloats,
      Coordinates2d.VIEW,
      convertFloatsOut
    )

    // Conduct a hit test using the VIEW coordinates
    val hits = frame.hitTest(convertFloatsOut[0], convertFloatsOut[1])
    val result = hits.getOrNull(0) ?: return null
    return result.trackable.createAnchor(result.hitPose)
  }
}

data class ARLabeledAnchor(val anchor: Anchor, val label: String)