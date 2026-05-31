from packages.simulation.CO import (
    State,
    StateDot,
    MeasuredState,
    NoiseForce,
    BacklashModel,
    ObjectOfControl,
    NoiseGenerator,
    SensorBlock,
)

# Проверка импорта — создаём экземпляры основных классов
print("=== Импорт OK ===")

# ObjectOfControl — однозвенный режим
config_pendulum = {
    "M": 1.0,
    "m1": 0.3,
    "m2": 0.0,
    "l1": 1.0,
    "l2": 0.0,
    "L1": 0.7,
    "L2": 0.0,
    "J1": 0.02,
    "J2": 0.0,
    "single_pendulum_mode": True,
    "backslash_mode": False,
    "init_q": [0.0, 0.15, 0.0],
}
plant = ObjectOfControl(config_pendulum)
print(f"ObjectOfControl: q={plant.q}, dq={plant.dq}")

# SensorBlock
config_sensor = {
    "encoder_resolution_1": 4096,
    "encoder_resolution_2": 4096,
    "cart_sensor_resolution": 0.0001,
    "noise_std_q": [0.001, 0.005, 0.005],
    "noise_std_dq": [0.01, 0.02, 0.02],
    "seed": 42,
}
sensor = SensorBlock(config_sensor)
raw_q = plant.q
raw_dq = plant.dq
measured = sensor.get_telemetry(raw_q, raw_dq)
print(f"SensorBlock: measured_state={measured}")

# BacklashModel
bl = BacklashModel(alpha=0.001, m_mot=0.1)
F_real = bl.update(F_ideal=10.0, cart_velocity=0.0, dt=0.001)
print(f"BacklashModel: F_real={F_real}, in_contact={bl.in_contact}")

# NoiseGenerator
ng = NoiseGenerator(std=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], seed=0)
noise = ng.generate()
print(f"NoiseGenerator: noise={noise}")

# Dataclasses
s = State(x=0.1, theta1=0.2, theta2=0.3)
sd = StateDot(x_dot=0.4, theta1_dot=0.5, theta2_dot=0.6)
ms = MeasuredState(x=0.1, theta1=0.2, theta2=0.3, x_dot=0.4, theta1_dot=0.5, theta2_dot=0.6)
nf = NoiseForce(value=1.5)
print(f"State: {s}")
print(f"StateDot: {sd}")
print(f"MeasuredState: {ms}")
print(f"NoiseForce: {nf}")

print("\n=== Все тесты пройдены ===")
