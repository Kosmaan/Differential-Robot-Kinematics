import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class SCR(Node):
    def __init__(self):
        super().__init__("scr_node")

        self.velocity_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.laser_subscriber = self.create_subscription(
            LaserScan, "/laser", self.laser_callback, 10
        )
        self.timer_handler = self.create_timer(0.05, self.kinematics_callback)

        # Constantele initiale
        self.k_ro = 1.0
        self.k_beta = -0.5
        self.k_alfa = 2.0
        self.timestep = 0.06

        self.x_current = 0.0
        self.y_current = 0.0
        self.theta_current = np.pi / 6

        # Pozitia finala
        self.x_target = 10.0
        self.y_target = 0.0
        self.theta_target = -np.pi / 6

        # Constante pentru evitarea obstacolelor
        self.min_safe_distance = 0.5
        self.laser_ranges = None
        self.obstacle_detected = False

    # Metoda pentru procesarea datelor de la LiDAR scan
    def laser_callback(self, msg: LaserScan):
        if not msg.ranges or len(msg.ranges) != 16:
            self.get_logger().error("Unexpected LiDAR data format!")
            return

        ranges = np.array(msg.ranges)
        ranges = np.where(
            (ranges < msg.range_min) | (ranges > msg.range_max),
            msg.range_max + 1,
            ranges,
        )
        # Urmarim doar laserele din fata (0), inainte stanga(1) si inainte dreapta(15)
        self.laser_ranges = ranges
        front_distances = [ranges[i] for i in [0, 1, 15]]

        # Verificam daca orice laser urmarit de noi a detectat un obstacol mai aproape de distanta impusa
        self.obstacle_detected = any(distance < self.min_safe_distance for distance in front_distances)

        if self.obstacle_detected:
            self.get_logger().info(f"Obstacle detected! Distances: {', '.join([f'{d:.2f}' for d in front_distances])}")

    def kinematics_callback(self):
        # Actualizeaza pozitia si viteza robotului pe baza modelului cinematic
        if self.obstacle_detected:
            self.avoid_obstacle()
            return

        dx = self.x_target - self.x_current
        dy = self.y_target - self.y_current
        ro = np.sqrt(dx**2 + dy**2)

        # Conditiile de oprire a robotului
        if ro < 0.1 and abs(self.theta_target - self.theta_current) < 0.1:
            self.get_logger().info("Robot has reached the target.")
            self.publish_stop()
            self.timer_handler.cancel()
            return

        alfa = np.arctan2(dy, dx) - self.theta_current
        beta = self.theta_target - self.theta_current

        v = self.k_ro * ro
        w = self.k_alfa * alfa + self.k_beta * beta

        # Limitam vitezele astfel incat sa nu isi ia zborul robotul
        v = min(v, 0.5)
        w = np.clip(w, -0.5, 0.5)

        self.theta_current += self.timestep * w
        self.x_current += self.timestep * v * np.cos(self.theta_current)
        self.y_current += self.timestep * v * np.sin(self.theta_current)

        # Cream variabila de tip Twist si o publicam pentru a actualiza vitezele robotului
        twist_msg = Twist()
        twist_msg.linear.x = v
        twist_msg.angular.z = w
        self.velocity_publisher.publish(twist_msg)

        self.get_logger().info(f"x: {self.x_current:.2f}")

    # Metoda pentru evitarea obstacolelor
    def avoid_obstacle(self):
        # Cream variabile pentru a sti in ce directii se afla obstacolele
        if self.laser_ranges is not None:
            left_clear = self.laser_ranges[1] > self.min_safe_distance
            right_clear = self.laser_ranges[15] > self.min_safe_distance
            forward_clear = self.laser_ranges[0] > self.min_safe_distance

            # Logica pentru evitarea obstacolelor
            if not forward_clear:
                self.publish_velocity(0.0, 0.5)
                self.get_logger().info("Obstacle in front. Rotating left.")
                return
            elif not left_clear and not right_clear:
                self.publish_velocity(-0.2, 0.0)
                self.get_logger().info("Obstacle on the left and right. Moving backwards")
                return
            elif not left_clear and not forward_clear:
                self.publish_velocity(0.0, -0.5)
                self.get_logger().info("Obstacle on the left. Rotating right.")
                return
            elif not right_clear and not forward_clear:
                self.publish_velocity(0.0, 0.5)
                self.get_logger().info("Obstacle on the right. Rotating left.")
                return
            elif not left_clear:
                self.publish_velocity(0.0, -0.5)
                self.get_logger().info("Obstacle on the left. Rotating right.")
                return
            elif not right_clear:
                self.publish_velocity(0.0, 0.5)
                self.get_logger().info("Obstacle on the right. Rotating left.")
                return
            
    # Metoda pentru actualizarea vitezelor robotului pentru modularitate
    def publish_velocity(self, linear, angular):
        twist_msg = Twist()
        twist_msg.linear.x = linear
        twist_msg.angular.z = angular
        self.velocity_publisher.publish(twist_msg)

    # Metoda pentru oprirea robotului cand ajunge la destinatie
    def publish_stop(self):
        self.publish_velocity(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    scr = SCR()
    rclpy.spin(scr)
    scr.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
