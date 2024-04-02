import numpy as np

#her bir T anında alınan ölçüm sayısı
numMeasPerInst = 20

#simülasyon sayısı
n_simulation = 7

#hız
speed = 0.5

#hedef pozisyonunun başlangıcı
target_initial_position = np.array([10., 10])

#hedef çember 
radius = 4.0

#sensör pozisyonu
sensor_location = [0.0, 0.0]

#örnekleme zamanı
T = 20

#ölçümleri rastgele açıdan al
select_random_theta = False

#ölçüm gürültüsü
stdMeas = 0.0

# manevra ilişki sabiti
theta_maneuver = 0.1

#ivmelenme süresi (genişliği)
accel_width = 0.1

#zamansal bozunma sabiti. tao = sonsuz -> hedef statik kabul edilir.
tao = 100 #100 adımda sistemin ömrü yaklaşık 0.3679'a düşüyor. Sistem zamanla bozuluyor.

#çizim için limit ayarları 
xlim = [-5., 40]
ylim = [-15, 25]
