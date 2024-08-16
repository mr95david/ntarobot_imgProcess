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
from yolov8_msgs.msg import DetectionArray, BoundingBox2D, Mask, Point2D, KeyPoint2DArray, KeyPoint2D, Detection
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
# Seccion de importe de protocolos de comunicacion de ros
from rclpy.qos import QoSReliabilityPolicy, QoSDurabilityPolicy, QoSProfile, QoSHistoryPolicy
# Importe de librerias utilitarias
from cv_bridge import CvBridge
import torch
from typing import List, Dict
import os

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
                #("topic_img", "/color/image_raw"), # Nombre de topico a leer de la imagen
                ("topic_img", "/image_raw"),
                ("model", "./src/ntarobot_camera/models/yolov8n.pt"), # Modelo de ejecucion
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
        self.get_logger().info(f"Activating {self.get_name()}")

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
        self.enable = bool(request.data)
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

            # Definicion de ejecucion de cpu
            results: Results = results[0].cpu()

            # Condicional de descripcion de cajas de los resultados obtenidos, sea orientados o estaticos
            if results.boxes or results.obb:
                # Se aplica una funcion de desarrollo de hypotesis de prediccion
                hipotesis = self.parse_hypo(results)
                boxes = self.parse_boxes(results)

            # En caso de obtener valores de mascara se realiza el condicional correspondiente
            if results.masks:
                # Se almacena la lista de mascaras obtenidas
                masks = self.parse_mask(results)
            
            #finalmente para los puntos llave de los objetos
            if results.keypoints:
                keypoints = self.parse_keypoints(results)
            
            # create detection msgs
            detections_msg = DetectionArray()

            for i in range(len(results)):

                aux_msg = Detection()

                if results.boxes or results.obb:
                    aux_msg.class_id = hipotesis[i]["class_id"]
                    aux_msg.class_name = hipotesis[i]["class_name"]
                    aux_msg.score = hipotesis[i]["score"]

                    aux_msg.bbox = boxes[i]

                if results.masks:
                    aux_msg.mask = masks[i]

                if results.keypoints:
                    aux_msg.keypoints = keypoints[i]

                detections_msg.detections.append(aux_msg)

            # publish detections
            detections_msg.header = msg.header
            self._pub.publish(detections_msg)

            del results
            del cv_img

    # Propuesta de funcion de ejecucion de hipotesis
    def parse_hypo(self, results: Results) -> list[Dict]:
        # La salida corresponde a una lista de caracteristicas
        hipo_list = []

        # Condicional de validacion de existencia de cajas en la prediccion
        if results.boxes:
            # Definicion de variable que almacena las cajas obtenidas
            box_data: Boxes
            # Ciclo para procesar cada una de esas marcaciones
            for box_data in results.boxes:

                # Desarrollo de la hipotesis - Caracteristicas de valor de los objetos encontrados
                hipo = {
                    "class_id": int(box_data.cls), # Id de objeto segun lista coco
                    "class_name": self.yolo.names[int(box_data.cls)], # Nombre de objeto
                    "score": float(box_data.conf) # Puntaje de aproximacion
                }
                # Agregar el objeto a la lista
                hipo_list.append(hipo)

        # Orientacion de objetos encontrados
        elif results.obb:
            # Ciclo por las coordenadas de los objetos
            for i in range(results.obb.cls.shape[0]):
                # Actualizacion de informacion de los nombres de yolo
                hipo = {
                    "class_id": int(results.obb.cls[i]),
                    "class_name": self.yolo.names[int(results.obb.cls[i])],
                    "score": float(results.obb.conf[i])
                }
                hipo_list.append(hipo)

        return hipo_list

    # Funcion para traducir la informacion de las cajas en posiciones y visualizacion para la imagen
    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:

        # Lista de cajas encontrados en los resultados
        boxes_list = []

        # Condicional de validacion de existencia de cajas
        if results.boxes:
            # variable por cada caja
            box_data: Boxes
            # Ciclo encontrada para cada caja en la lista obtenida de los resultados
            for box_data in results.boxes:
                # Asignacion de valor a msg prediseñado
                msg = BoundingBox2D()

                # Obtencion de valores de caja (Coordenadas)
                box = box_data.xywh[0]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])

                # Agregado de mensaje referente a informacion de cajas en la lista de salida
                boxes_list.append(msg)

        # Para el caso de los valores de orientacion de las cajas se obtiene el ciclo
        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                # y se almacenan los datos de cajas en un mensaje especifco
                msg = BoundingBox2D()

                # get boxes values
                box = results.obb.xywhr[i]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.center.theta = float(box[4])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])

                # append msg
                boxes_list.append(msg)

        return boxes_list

    # La funcion de obtencion de mascaras, sera dada por la entrada del argumento de resultados
    def parse_masks(self, results: Results) -> List[Mask]:

        # See crea la lista de mascaras de salida
        masks_list = []

        # funcion de scope local para, Creacion de punto 2d en una posicion especifica
        def create_point2d(x: float, y: float) -> Point2D:
            # mensaje especifico para los datoss
            p = Point2D()
            p.x = x
            p.y = y
            return p

        # Designacion de mensajes de salida de mascara
        mask: Masks
        # Ciclo para leer cada valor de la lista general
        for mask in results.masks:
            # Creaccion de mensaje preconfigurado para determinar mascara
            msg = Mask()
            # Asignar valores obtenidos a la lista
            msg.data = [create_point2d(float(ele[0]), float(ele[1]))
                        for ele in mask.xy[0].tolist()]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]
            # Generacion de lista de mensajes obtenidos
            masks_list.append(msg)
        # Retorno de lista final
        return masks_list

    # Finalmente la funcion de puntos importantes
    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:
        # Lista de putos de importancia
        keypoints_list = []

        # Definicion de variable de almacenamiento
        points: Keypoints
        # ciclo por cada uno de los puntos encontrados
        for points in results.keypoints:
            # Creacion de mensaje especifico
            msg_array = KeyPoint2DArray()
            # Validacion de exs}istencia de puntos
            if points.conf is None:
                continue
            # Controlador de posicion de puntos
            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):
                # Validacion de thresh de los puntos validados
                if conf >= self.threshold:
                    msg = KeyPoint2D()

                    msg.id = kp_id + 1
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)

                    msg_array.data.append(msg)
            # Lista obtenida de los arrays
            keypoints_list.append(msg_array)
        # retorno final de valores
        return keypoints_list
    


def main():
    rclpy.init()

    move_pose = CameraNode()
    rclpy.spin(move_pose)
    
    move_pose.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()