from datetime import datetime
import numpy as np
from orbit import ISS
from skyfield.api import load
from time import time

ISS = ISS()


def main():
    linear_velocities = []

    t = load.timescale().now()
    days_since_update = t - ISS.epoch
    print(f"{days_since_update} days since epoch")
    start: float = time()

    while (time() - start) < 598:
        now = datetime.now()
        ts = load.timescale()
        t = ts.utc(now.year, now.month, now.day, now.hour, now.minute)

        barycentric = ISS.at(t)
        velocity_vector = barycentric.velocity.km_per_s
        linear_velocity = np.linalg.norm(velocity_vector)
        linear_velocities.append(linear_velocity)
        print(np.mean(linear_velocities))

    with open("results.txt", "w") as results_txt:
        results_txt.write(f"{np.mean(linear_velocities):.5f}")


if __name__ == "__main__":
    main()