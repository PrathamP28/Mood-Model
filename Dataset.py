import random
import pandas as pd

data = []

for _ in range(500):
    temp = random.randint(15, 40)
    humidity = random.randint(30, 90)
    rain = random.randint(0, 2)
    sleep = random.randint(3, 9)
    steps = random.randint(1000, 12000)
    hour = random.randint(0, 23)
    season = random.randint(0, 2)

    # RULES (your logic)
    mood = 50
    mood += sleep * 5
    mood += steps / 2000
    mood -= rain * 10
    mood -= (humidity - 60) * 0.2
    mood += (25 - abs(temp - 25)) * 0.5

    mood = max(0, min(100, mood))

    data.append([temp, humidity, rain, sleep, steps, hour, season, mood])

df = pd.DataFrame(data, columns=[
    "temp","humidity","rain","sleep","steps","hour","season","mood"
])

df.to_csv("data.csv", index=False)
