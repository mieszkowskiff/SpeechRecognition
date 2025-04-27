from PIL import Image

# Lista ścieżek do obrazków
filenames = [f"./images/0.png", f"./images/0.png"] + [f"./images/{i}.png" for i in range(11)]

# Wczytaj wszystkie obrazki
frames = [Image.open(fn) for fn in filenames]

# Zapisz jako GIF
frames[0].save(
    './images/output.gif',
    format='GIF',
    append_images=frames[1:],  # kolejne klatki
    save_all=True,
    duration=800,  # czas trwania jednej klatki (w ms)
    loop=0  # 0 = zapętlenie nieskończone
)