#!/usr/bin/env python3

# Código modificado a partir de algoritmo tomado de github
# Fuente: https://github.com/mgonzs13/yolov8_ros/tree/main
# Autor: Miguel Ángel González Santamarta - @mgonzs13 

# Seccion de importe de librerias
# Librerias de ejecucion de ros
import rclpy
# Librerias de nodo
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState
# Librerias de protcolo de comunicacion
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
# Librerias de procesamiento de imagen
from ultralytics.engine.results import Boxes
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml
# Librerias utilitarias
import numpy as np
import message_filters
from cv_bridge import CvBridge
# Librerias de interfaz
from sensor_msgs.msg import Image
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray

# Creacion de clase de nodo de lifecycle
class TrackImgNode(LifecycleNode):
    # Constructor de clase
    def __init__(self) -> None:
        # Nombre de nodo
        super().__init__("tracking_node")

        # Ingreso de parametros
        self.declare_parameter("tracker", "bytetrack.yaml")
        self.declare_parameter("image_reliability",
                               QoSReliabilityPolicy.BEST_EFFORT)

        # Creacion de variable de comunicacion
        self.cv_bridge = CvBridge()
        
        # Validacion visual de ejecucion de nodo
        self.get_logger().info("Tracking node inicialized")

    # Funcion de configuracion de nodo
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        # Validacion de inicio de configuracion de nodo
        self.get_logger().info(f"Configuring {self.get_name()}")

        # Parametro de nombre de tracker
        tracker_name = self.get_parameter(
            "tracker").get_parameter_value().string_value
        # Protocolo para prioridad de comunicacion  de imagen
        self.image_reliability = self.get_parameter(
            "image_reliability").get_parameter_value().integer_value
        # Creacion de tracker de objetos
        self.tracker = self.create_tracker(tracker_name)
        # Crecacion de publicador de detector
        self._pub = self.create_publisher(DetectionArray, "tracking", 10)

        # Retorno de estado de nodo
        return TransitionCallbackReturn.SUCCESS
    
    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Activating {self.get_name()}")

        image_qos_profile = QoSProfile(
            reliability=self.image_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # subs
        image_sub = message_filters.Subscriber(
            self, Image, "/color/image_raw", qos_profile=image_qos_profile)
        detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections", qos_profile=10)

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (image_sub, detections_sub), 10, 0.5)
        self._synchronizer.registerCallback(self.detections_cb)

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Deactivating {self.get_name()}")

        self.destroy_subscription(self.image_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)

        del self._synchronizer
        self._synchronizer = None

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Cleaning up {self.get_name()}")

        del self.tracker

        return TransitionCallbackReturn.SUCCESS

    def create_tracker(self, tracker_yaml: str) -> BaseTrack:

        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        check_requirements("lap")  # for linear_assignment

        tracker = check_yaml(tracker_yaml)
        cfg = IterableSimpleNamespace(**yaml_load(tracker))

        assert cfg.tracker_type in ["bytetrack", "botsort"], \
            f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=1)
        return tracker

    def detections_cb(self, img_msg: Image, detections_msg: DetectionArray) -> None:

        tracked_detections_msg = DetectionArray()
        tracked_detections_msg.header = img_msg.header

        # convert image
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)

        # parse detections
        detection_list = []
        detection: Detection
        for detection in detections_msg.detections:

            detection_list.append(
                [
                    detection.bbox.center.position.x - detection.bbox.size.x / 2,
                    detection.bbox.center.position.y - detection.bbox.size.y / 2,
                    detection.bbox.center.position.x + detection.bbox.size.x / 2,
                    detection.bbox.center.position.y + detection.bbox.size.y / 2,
                    detection.score,
                    detection.class_id
                ]
            )

        # tracking
        if len(detection_list) > 0:

            det = Boxes(
                np.array(detection_list),
                (img_msg.height, img_msg.width)
            )

            tracks = self.tracker.update(det, cv_image)

            if len(tracks) > 0:

                for t in tracks:

                    tracked_box = Boxes(
                        t[:-1], (img_msg.height, img_msg.width))

                    tracked_detection: Detection = detections_msg.detections[int(
                        t[-1])]

                    # get boxes values
                    box = tracked_box.xywh[0]
                    tracked_detection.bbox.center.position.x = float(box[0])
                    tracked_detection.bbox.center.position.y = float(box[1])
                    tracked_detection.bbox.size.x = float(box[2])
                    tracked_detection.bbox.size.y = float(box[3])

                    # get track id
                    track_id = ""
                    if tracked_box.is_track:
                        track_id = str(int(tracked_box.id))
                    tracked_detection.id = track_id

                    # append msg
                    tracked_detections_msg.detections.append(tracked_detection)

        # publish detections
        self._pub.publish(tracked_detections_msg)


# Funcion de ejecucion del nodo
def main():
    rclpy.init()

    track_node = TrackImgNode()
    rclpy.spin(track_node)
    track_node.trigger_configure()
    track_node.trigger_activate()
    track_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()