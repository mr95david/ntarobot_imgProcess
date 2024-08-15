#!/usr/bin/env python3

# El siguiente nodo corresponde a una prueba de ejecucion con la imagen proveniente de una imagen
# de la camara realsense 515, de tal manera se define tambien que este puede ser ejecutado directamente
# en la maquina de supervision, como un eje maestro.

# Seccion de importe de librerias
# Importe de librerias de ros
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
# Seccion de importe de librerias de yolo
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes, Masks, Keypoints
# Seccion de importe de librerias de interfaces
from yolov8_msgs.msg import DetectionArray
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
# Seccion de importe de protocolos de comunicacion de ros
from rclpy.qos import QoSReliabilityPolicy, QoSDurabilityPolicy, QoSProfile, QoSHistoryPolicy
# Importe de librerias utilitarias
from cv_bridge import CvBridge
import torch

# Creacion de clase de nodo
# Para este caso realizaremos un nodo con un control de ciclo de vida, con tal de establecer los momentos de uso de
# la funcionalidad deseada
class CameraNode(LifecycleNode):
    # Consturctor
    def __init__(self) -> None:
        # herencia de nodo
        super().__init__("camera_comm_node")

        # Declaracion de parametros
        self.declare_parameters(
            namespace = "",
            parameters = [
                ("topic_img", "/color/image_raw"), # Nombre de topico a leer de la imagen
                ("model", "yolo8n.pt"), # Modelo de ejecucion
                ("device", "cuda:0"), # Dispositivo, cpu o gpu
                ("threshold", 0.5),
                ("enable", True),
                ("image_reliability", QoSReliabilityPolicy.BEST_EFFORT),
            ]
        )

        # Validacion de inicio de nodo
        self.get_logger().info("Image General Process Created")

    # Continua la ejecucion de las funciones de configuracion del nodo
    # Configuracion de configuracion de nodo
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        # Validacion de configuracion de nodo
        self.get_logger().info(f"Configuring {self.get_name()}")

        # Para la configuracion es necesario crear todas las variables de instancia relacionadas a los parametros iniciales
        self._init_vars()

        # Inicializacion de publifadores
        self._pub = self.create_lifecycle_publisher(
            DetectionArray, # Tipo de mensaje usado
            "/detections", # Nombre de topico
            10 # Protocolo QoS
        )

        # Inicializacion de servicio
        self._srv = self.create_service(
            SetBool, # Tipo de mensaje para el servicio
            "enable", # Nombre de servicio
            self.callBackEnable # Funcion de callback del servicio
        )

        # Retorno de ejecucion de proceso de configuracion de lifecycle
        return TransitionCallbackReturn.SUCCESS
    
    # Funcion de activacion de nodo
    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        # Mensaje de validacion de activacion
        self.get_logger().info("Activating {self.get_name()}")

        # Inicializacion de YOLO : NOTA: Es necesario definir si se desea una ruta especifica del modelo de uso de yolo
        self.yolo = YOLO(self.model)

        # Configuracion de subscriptor de la imagen
        self._sub_img = self.create_subscription(
            Image,
            self.topic_img,
            self.callBackImage, # Callback de imagen obtenida
            self.image_qos_profile # Protocolo de comunicacion
        )

        super().on_activate(state)

        # Proceso ejecutado con exito
        return TransitionCallbackReturn.SUCCESS
    
    # Funcion de desactivacion de nodo con cilco de vida
    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        # Validacion de ejecucion de status
        self.get_logger().info(f"Deactivating {self.get_name()}")

        # Se elimina la instancia creada de yolo
        del self.yolo

        # Validacion de uso de gpu
        if "cuda" in self.device:
            # Limpieza de basura y cache
            self.get_logger().info("Limpiando cache de la gpu")

            # Usamos la funcion  de torch destinada para ese objetivo
            torch.cuda.empty_cache()

        # Destruye tambien el subscriptor creado
        self.destroy_subscription(self._sub_img)
        # Limpieza de variable
        self._sub_img = None

        super().on_deactivate(state)

        # Proceso ejecutado con exito
        return TransitionCallbackReturn.SUCCESS
    
    # Funcion para limpieza general de nodo
    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        # Validacion de limpiza de nodo
        self.get_logger().info(f"Cleaning up {self.get_name()}")

        # Destruccion de publicador
        self.destroy_publisher(self._pub)

        # Borrado de protocolo de comunicacion 
        del self.image_qos_profile

        return TransitionCallbackReturn.SUCCESS

    # Seccion de funciones generales
    # Funcion de inicializacion de variables de instancia
    def _init_vars(self) -> None:
        # Para la configuracion es necesario crear todas las variables de instancia relacionadas a los parametros iniciales
        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        self.reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value
        # topic de imagen
        self.topic_img = self.get_parameter("topic_img").get_parameter_value().string_value

        # Definicion de perfil de QoSProfile
        self.image_qos_profile = QoSProfile(
            reliability = self.reliability,
            history = QoSHistoryPolicy.KEEP_LAST,
            durability = QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )

        # Variable de cv_bridge
        self.cv_bridge_var = CvBridge()

    # Funcion de callback de servicio
    def callBackEnable(self, request, response):
        # Data de solicitud
        self.enable = request.data
        # Respuesta de suceso
        response.success = True

        # Retorno final
        return response

    # Seccion de funciones de callback de imagen
    def callBackImage(self, msg: Image) -> None:
        # Validacion de ejecucion de nodo por medio de servicio
        if self.enable:

            # Realizar conversion de imagen
            cv_img = self.cv_bridge_var.imgmsg_to_cv2(msg)

            # Se predicen los resultados de los objetos en la imagen
            results = self.yolo.predict(
                source = cv_img, # Corresponde a la imagen que se quiere procesar
                verbose = False,
                stream = False,
                conf = self.threshold, # configuracion predeterminada
                device = self.device # Tipo de procesamiento efectuado
            )

            # 

def main():
    rclpy.init()

    move_pose = CameraNode()
    rclpy.spin(move_pose)
    
    move_pose.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()