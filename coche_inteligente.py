import math
import sys
import os
import datetime
import pygame
import numpy as np
import cv2
import random
import   pandas as pd 
import joblib
sys.stdout = sys.__stdout__
# Constants
WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 60
CAR_SIZE_Y = 60
BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit

MANUAL_MODE = False  
AUTO_MODE = False
IA_MODE = True      


MANUAL_ARFF = 'driving_data_auto_manual.arff'
MAPA_PATH = 'test/map3.png' 

#########################################################################################################################################################################################
def encontrar_posicion_coche_en_linea_roja(ruta_mapa, car_size_x=60, car_size_y=60):
    """
    Busca una línea roja (no solo el rojo puro) en la imagen y coloca el coche en el centro de la línea.
    """
    img = cv2.imread(ruta_mapa)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_mapa}")
    # Usar rango de rojo en BGR (OpenCV)
    lower_red = np.array([0, 0, 120])    # B, G, R
    upper_red = np.array([80, 80, 255])
    mask = cv2.inRange(img, lower_red, upper_red)
    puntos = np.argwhere(mask == 255)
    if len(puntos) == 0:
        raise RuntimeError("No se encontró línea roja en el mapa")
    # Centro de la línea
    y, x = np.mean(puntos, axis=0).astype(int)
    car_x = int(x - car_size_x / 2)
    car_y = int(y - car_size_y / 2)
    return [car_x, car_y]


#######################################################################################################################################################################################


class Car:
    def __init__(self, sprite_path='car.png', speed=8, mapa_path=MAPA_PATH):
        self.position = encontrar_posicion_coche_en_linea_roja(mapa_path)
        self.sprite = pygame.image.load(sprite_path).convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite
        self.angle = random.randint(0, 360)
        self.speed = speed
        self.alive = True
        self.lives = 3
        self.center = [0, 0]
        self.radars = []
        self.last_error = 0
        self.last_steering = 0
        self.distance = 0
        self.time = 0
        self.update_center()
        print(f"Posición inicial del coche: {self.position}")

    def update_center(self):
        self.center = [
            self.position[0] + CAR_SIZE_X / 2,
            self.position[1] + CAR_SIZE_Y / 2
        ]

    def rotate_center(self, image, angle):
        rect = image.get_rect()
        rotated = pygame.transform.rotate(image, angle)
        new_rect = rect.copy()
        new_rect.center = rotated.get_rect().center
        return rotated.subsurface(new_rect).copy()

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)
        for (x, y), _ in self.radars:
            pygame.draw.line(screen, (0, 255, 0), self.center, (x, y), 1)
            pygame.draw.circle(screen, (0, 255, 0), (x, y), 5)

    def _check_collision(self, game_map):
        check_points = []
        for angle in range(0, 360, 45):
            length = 0.4 * min(CAR_SIZE_X, CAR_SIZE_Y)
            rad_angle = math.radians(360 - (self.angle + angle))
            x = int(self.center[0] + math.cos(rad_angle) * length)
            y = int(self.center[1] + math.sin(rad_angle) * length)
            check_points.append((x, y))
        check_points.append((int(self.center[0]), int(self.center[1])))
        collision_count = 0
        for x, y in check_points:
            if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
                collision_count += 1
            else:
                try:
                    if game_map.get_at((x, y)) == BORDER_COLOR:
                        collision_count += 1
                except IndexError:
                    collision_count += 1
        if collision_count > 0:
            self.lives -= 1
            if self.lives <= 0:
                self.alive = False
            else:
                self._reposition_car()

    def _reposition_car(self):
        try:
            print("Reposicionando coche...")
            new_pos = encontrar_posicion_coche_en_linea_roja(MAPA_PATH)
            self.position = new_pos
            self.angle = random.randint(0, 360)
            self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
            self.update_center()
            print(f"Nueva posición: {self.position}")
        except Exception as e:
            print(f"Error al reposicionar el coche: {e}")
            self.alive = False

    def _scan_radars(self, game_map, angles):
        self.radars.clear()
        for deg in angles:
            length = 0
            x = int(self.center[0])
            y = int(self.center[1])
            absolute_angle = 360 - (self.angle + deg)
            rad_angle = math.radians(absolute_angle)
            while length < 300:
                x = int(self.center[0] + math.cos(rad_angle) * length)
                y = int(self.center[1] + math.sin(rad_angle) * length)
                if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
                    break
                try:
                    if game_map.get_at((x, y)) == BORDER_COLOR:
                        break
                except IndexError:
                    break
                length += 1
            dist = int(math.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2))
            self.radars.append(((x, y), dist))

    def base_move_and_sense(self, game_map, steering, radar_angles):
        self.angle = (self.angle + steering) % 360
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        rad_angle = math.radians(360 - self.angle)
        dx = math.cos(rad_angle) * self.speed
        dy = math.sin(rad_angle) * self.speed
        prev_position = self.position.copy()
        self.position[0] += dx
        self.position[1] += dy
        self.update_center()
        self._check_collision(game_map)
        if self.alive:
            self.distance += math.sqrt(dx ** 2 + dy ** 2)
            self.time += 1
        else:
            self.position = prev_position
            self.update_center()
        self._scan_radars(game_map, radar_angles)

    def get_manual_row(self, left, right):
        sensors = [d for _, d in self.radars]
        return sensors + [int(left), int(right)]

    def get_auto_row(self):
        left = self.radars[0][1]
        front = self.radars[1][1]
        right = self.radars[2][1]
        steering = round((self.angle % 360) - 180, 2)
        return [left, front, right, steering, round(self.speed, 2)]

def save_regression_data(arff_file, sensor_values, steering):
    if not os.path.exists(arff_file):
        with open(arff_file, 'w') as f:
            f.write("% Datos de regresión de conducción (manual + auto)\n")
            f.write("@RELATION driving_regression\n")
            f.write("@ATTRIBUTE s_left   REAL\n")
            f.write("@ATTRIBUTE s_lc     REAL\n")
            f.write("@ATTRIBUTE s_c      REAL\n")
            f.write("@ATTRIBUTE s_rc     REAL\n")
            f.write("@ATTRIBUTE s_right  REAL\n")
            f.write("@ATTRIBUTE steering REAL\n")
            f.write("@DATA\n")
    with open(arff_file, 'a') as f:
        fila = list(map(str, sensor_values)) + [str(steering)]
        f.write(','.join(fila) + "\n")

######################################################################################################################################################################################
#Coche modo manual
class ManualCar(Car):
    def __init__(self):
        super().__init__(speed=10)
    def update(self, game_map, steering, left_flag, right_flag):
        self.base_move_and_sense(game_map, steering + self.centering_error(), [-90, -45, 0, 45, 90])
    def centering_error(self):
        if len(self.radars) < 5:
            return 0
        left = self.radars[0][1]
        right = self.radars[4][1]
        error = right - left
        corr = error * 0.02 + (error - self.last_error) * 0.1
        self.last_error = error
        return corr

######################################################################################################################################################################################
class AutoCar(Car):
    def __init__(self):
        super().__init__(speed=8)
        self.kp = 0.1
        self.ki = 0.001
        self.kd = 0.05
        self.integral = 0.0
        self.last_error = 0.0
        self.max_speed = 12
        self.min_speed = 4
        self.front_threshold = 60
        self.steering_history = []
    

    def decide(self):
        if not self.alive or len(self.radars) < 5:
            return 0.0
        d0, d1, d2, d3, d4 = [d for _, d in self.radars]
        left_side = (d0 + d1) / 2
        right_side = (d3 + d4) / 2
        error = right_side - left_side
        self.integral += error
        derivative = error - self.last_error
        self.last_error = error
        corr = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.steering_history.append(corr)
        if len(self.steering_history) > 5:
            self.steering_history.pop(0)
        smoothed_steering = sum(self.steering_history) / len(self.steering_history)
        return max(min(smoothed_steering, 5.0), -5.0)

    def update(self, game_map):
        angles = [-90, -45, 0, 45, 90]
        self._scan_radars(game_map, angles)
        steering = self.decide()
        self.last_steering = steering
        deadzone = 1.0
        if abs(steering) < deadzone:
            steering = 0.0
        front = self.radars[2][1]
        if front < self.front_threshold:
            self.speed = max(self.min_speed, self.speed - 0.5)
        else:
            self.speed = min(self.max_speed, self.speed + 0.2)
        self.base_move_and_sense(game_map, steering, angles)


#########################################################################################################################################################################################

class SmartCar(Car):
    def __init__(self, 
                 regressor_path='modelo_regresion.joblib', 
                 classifier_path='modelo_clasificador.joblib',
                 speed=8,
                 mapa_path=MAPA_PATH):
        super().__init__(speed=speed, mapa_path=mapa_path)
        # Carga modelos
        self.regressor = joblib.load(regressor_path) if regressor_path else None
        self.classifier = joblib.load(classifier_path) if classifier_path else None

        # Columnas esperadas (ajusta si tu modelo espera otro nombre de input)
        self.sensor_names = ['s_left', 's_lc', 's_c', 's_rc', 's_right']
        # Si guardas los nombres de columnas en el modelo, usa eso:
        if self.regressor and hasattr(self.regressor, 'feature_names_in_'):
            self.reg_columns = list(self.regressor.feature_names_in_)
        else:
            self.reg_columns = self.sensor_names
        if self.classifier and hasattr(self.classifier, 'feature_names_in_'):
            self.clf_columns = list(self.classifier.feature_names_in_)
        else:
            self.clf_columns = self.sensor_names

        # Si tu clasificador guarda el encoder de clases
        self.target_encoder = getattr(self.classifier, 'target_encoder', None)
        self.last_steering = 0

    def get_sensor_features(self):
        sensors = [d for _, d in self.radars]
        features = dict(zip(self.sensor_names, sensors))
        return features

    def steering_from_class(self, class_label):
        """Convierte la predicción de clase en steering. Ajusta según tu mapping real."""
        if isinstance(class_label, (np.ndarray, list)):
            class_label = class_label[0]
        if hasattr(class_label, 'decode'):  # Si viene como bytes
            class_label = class_label.decode()
        class_label = str(class_label).lower()
        if class_label in ['left', 'izquierda']:
            return 5.0
        elif class_label in ['right', 'derecha']:
            return -5.0
        elif class_label in ['straight', 'center', 'recto', 'centro', 'c']:
            return 0.0
        # Si la clase es un número (ej: -5, 0, 5)
        try:
            return float(class_label)
        except:
            return 0.0

    

    def decide(self):
        features = self.get_sensor_features()
        X_clf = pd.DataFrame([features], columns=self.clf_columns)
        X_reg = pd.DataFrame([features], columns=self.reg_columns)

        steering_reg = None
        steering_class = None

        # Usar regresión si existe
        if self.regressor:
            try:
                steering_reg = float(self.regressor.predict(X_reg)[0])
            except Exception as e:
                print("Error en predicción regresión:", e)
                steering_reg = None

        # Usar clasificación si existe
        if self.classifier:
            try:
                class_pred = self.classifier.predict(X_clf)[0]
                # Si hay target_encoder, transforma a etiqueta original
                if self.target_encoder:
                    class_pred = self.target_encoder.inverse_transform([class_pred])[0]
                steering_class = self.steering_from_class(class_pred)
            except Exception as e:
                print("Error en predicción clasificación:", e)
                steering_class = None

        # CASOS:
        # Ambos existen: promediar
        # Solo uno existe: usar ese
        if steering_reg is not None and steering_class is not None:
            steering = (steering_reg + steering_class) / 2.0
        elif steering_reg is not None:
            steering = steering_reg
        elif steering_class is not None:
            steering = steering_class
        else:
            steering = 0.0

        # Limitar steering igual que tus otros coches
        steering = max(min(steering, 5.0), -5.0)
        return steering





    def update(self, game_map):
        angles = [-90, -45, 0, 45, 90]
        self._scan_radars(game_map, angles)
        steering = self.decide()
        self.last_steering = steering
        self.base_move_and_sense(game_map, steering, angles)



#####################################################################################################################################################################################

def run_manual_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("Arial", 30)
    try:
        game_map = pygame.image.load(MAPA_PATH).convert()
    except Exception as e:
        print(f"No se pudo cargar el mapa {MAPA_PATH} ({e})")
        return
    car = ManualCar()
    clock = pygame.time.Clock()
    samples = 0
    last_sample_time = 0
    sample_interval = 0.2
    arff = MANUAL_ARFF
    if not os.path.exists(arff):
        with open(arff, 'w') as f:
            f.write("% Manual driving data\n@RELATION driving_data_manual\n")
            f.write("@ATTRIBUTE s_left REAL\n@ATTRIBUTE s_lc REAL\n@ATTRIBUTE s_c REAL\n@ATTRIBUTE s_rc REAL\n@ATTRIBUTE s_right REAL\n@ATTRIBUTE left {0,1}\n@ATTRIBUTE right {0,1}\n@DATA\n")
    running = True
    start_time = pygame.time.get_ticks()
    while running:
        current_time = pygame.time.get_ticks()
        elapsed_seconds = (current_time - start_time) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                if e.key == pygame.K_r:
                    car = ManualCar()
                    samples = 0
                    start_time = pygame.time.get_ticks()
                if e.key == pygame.K_s:
                    sample_interval = 0 if sample_interval > 0 else 0.2
                    status = "MUESTREO DETENIDO" if sample_interval == 0 else "MUESTREO ACTIVO"
                    print(status)
        keys = pygame.key.get_pressed()
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        steer = (3 if left else 0) + (-3 if right else 0)
        car.update(game_map, steer, left, right)
        if car.alive and (left or right or (elapsed_seconds - last_sample_time > sample_interval)):
            if left or right or abs(car.last_error) > 5:
                row = car.get_manual_row(left, right)
                with open(arff, 'a') as f:
                    f.write(','.join(map(str, row)) + "\n")
                samples += 1
                last_sample_time = elapsed_seconds
                sensors = [d for _, d in car.radars]
                steering = car.last_error
                save_regression_data('driving_regression_manual_auto.arff', sensors, steering)
        screen.blit(game_map, (0, 0))
        car.draw(screen)
        status = f"MODO MANUAL - Vidas: {car.lives}  Muestras: {samples}  Error: {car.last_error:.1f}"
        screen.blit(font.render(status, True, (0, 0, 0)), (50, 50))
        if len(car.radars) >= 5:
            sensor_info = f"Sensores: L:{car.radars[0][1]} LC:{car.radars[1][1]} C:{car.radars[2][1]} RC:{car.radars[3][1]} R:{car.radars[4][1]}"
            screen.blit(font.render(sensor_info, True, (0, 0, 0)), (50, 90))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()
    print(f"Manual completado: {samples} muestras en {arff}")

def run_auto_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Modo Automático Mejorado")
    font = pygame.font.SysFont("Arial", 24)
    try:
        game_map = pygame.image.load(MAPA_PATH).convert()
    except Exception as e:
        print(f"No se pudo cargar el mapa {MAPA_PATH} ({e})")
        return
    car = AutoCar()
    clock = pygame.time.Clock()
    laps = dist_lap = samples = 0
    start_pos = tuple(car.position)
    start = datetime.datetime.now()
    arff = 'driving_data_auto_manual.arff'
    if not os.path.exists(arff):
        with open(arff, 'w') as f:
            f.write("% Auto driving data (manual format)\n")
            f.write("@RELATION driving_data_auto_manual\n\n")
            f.write("@ATTRIBUTE s_left   REAL\n")
            f.write("@ATTRIBUTE s_lc     REAL\n")
            f.write("@ATTRIBUTE s_c      REAL\n")
            f.write("@ATTRIBUTE s_rc     REAL\n")
            f.write("@ATTRIBUTE s_right  REAL\n")
            f.write("@ATTRIBUTE left     {0,1}\n")
            f.write("@ATTRIBUTE right    {0,1}\n")
            f.write("@DATA\n")
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                if e.key == pygame.K_r:
                    car = AutoCar()
                    laps = dist_lap = samples = 0
                    start = datetime.datetime.now()
        prev = car.position.copy()
        car.update(game_map)
        moved = math.hypot(car.position[0] - prev[0], car.position[1] - prev[1])
        dist_lap += moved
        if math.hypot(car.position[0] - start_pos[0], car.position[1] - start_pos[1]) < 80 and dist_lap > 3000:
            laps += 1
            dist_lap = 0
        if car.alive and len(car.radars) == 5:
            left_flag = 1 if car.last_error > 0 else 0
            right_flag = 1 if car.last_error < 0 else 0
            row = car.get_manual_row(left_flag, right_flag)
            with open(arff, 'a') as f:
                f.write(','.join(map(str, row)) + "\n")
            samples += 1
            sensors = [d for _, d in car.radars]
            steering = car.last_error
            save_regression_data('driving_regression_manual_auto.arff', sensors, steering)
        screen.fill((30, 30, 30))
        screen.blit(game_map, (0, 0))
        car.draw(screen)
        elapsed = datetime.datetime.now() - start
        mins, secs = divmod(elapsed.seconds, 60)
        info = f"Tiempo {mins:02d}:{secs:02d}  Vueltas {laps}  Vidas {car.lives}  Muestras {samples}"
        screen.blit(font.render(info, True, (255, 255, 255)), (30, 30))
        if car.alive and len(car.radars) >= 5:
            sensor_info = f"Sensores: L:{car.radars[0][1]} LC:{car.radars[1][1]} C:{car.radars[2][1]} RC:{car.radars[3][1]} R:{car.radars[4][1]}"
            steering_info = f"Steering: {car.last_steering:.2f}° Speed: {car.speed:.2f}"
            screen.blit(font.render(sensor_info, True, (255, 255, 255)), (30, 60))
            screen.blit(font.render(steering_info, True, (255, 255, 255)), (30, 90))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()
    print(f"Auto completado: {samples} muestras en {arff}")


def run_ia_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Modo IA")
    font = pygame.font.SysFont("Arial", 24)
    try:
        game_map = pygame.image.load(MAPA_PATH).convert()
    except Exception as e:
        print(f"No se pudo cargar el mapa {MAPA_PATH} ({e})")
        return
    car = SmartCar(regressor_path='modelo_regresion.joblib', classifier_path='modelo_clasificador.joblib')
    clock = pygame.time.Clock()
    running = True
    samples = 0
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                if e.key == pygame.K_r:
                    car = SmartCar(regressor_path='modelo_regresion.joblib', classifier_path=' modelo_clasificador.joblib')
        car.update(game_map)
        screen.fill((30, 30, 30))
        screen.blit(game_map, (0, 0))
        car.draw(screen)
        status = f"MODO IA - Vidas: {car.lives}  Error: {getattr(car, 'last_steering', 0):.2f}"
        screen.blit(font.render(status, True, (255, 255, 255)), (50, 50))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()



if __name__ == "__main__":
    if MANUAL_MODE:
        print("Iniciando modo MANUAL...")
        run_manual_simulation()
    elif AUTO_MODE:
        print("Iniciando modo AUTOMÁTICO...")
        run_auto_simulation()
    elif IA_MODE:
        print("Iniciando modo Agente con inteligencia")
        run_ia_simulation()
    else:
        print("Selecciona un modo de operación válido (MANUAL_MODE, AUTO_MODE o IA_MODE)")
