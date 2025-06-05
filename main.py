"""
Proje Adı: Mamdani Bulanık Çıkarım ile ALE Tahmini
Açıklama: Bu Python betiği, WSN veriseti ile 4 giriş özelliğinden (anchor ratio, range, density, iteration)
          Average Localization Error (ALE) tahmini yapmak üzere Mamdani Bulanık Mantık Modeli kullanır.
          İki farklı üyelik fonksiyonu (Üçgen & Gauss) ve iki farklı defuzzification yöntemi (Centroid & Weighted Avg.)
          ile dört farklı kombinasyon denenmiş ve MAE / RMSE ile performansları karşılaştırılmıştır.
"""

import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Veri setini oku
df = pd.read_csv("mcs_ds_edited_iter_shuffled.csv")

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]
y_true = y.values

# Normalize
X_normalized = (X - X.min()) / (X.max() - X.min())
y_min, y_max = y.min(), y.max()

# Evren
universe = np.linspace(0, 1, 100)
ale_range = universe

# Üyelik fonksiyonları
def create_tri_membership(range_):
    return {
        "low": fuzz.trimf(range_, [0, 0, 0.5]),
        "mid": fuzz.trimf(range_, [0, 0.5, 1]),
        "high": fuzz.trimf(range_, [0.5, 1, 1]),
    }

def create_gauss_membership(range_):
    return {
        "low": fuzz.gaussmf(range_, 0.2, 0.1),
        "mid": fuzz.gaussmf(range_, 0.5, 0.1),
        "high": fuzz.gaussmf(range_, 0.8, 0.1),
    }

# Ana çıkarım fonksiyonu
def run_fuzzy_inference(X_norm, anchor_mf, range_mf, density_mf, iteration_mf, ale_mf, defuzz_method='centroid', label=''):
    predictions = []

    for i in range(len(X_norm)):
        a, r, d, it = X_norm.iloc[i]
        rules = []

        def m(mf, key, val): return fuzz.interp_membership(universe, mf[key], val)

        rules.extend([
            (np.fmin(np.fmin(m(anchor_mf, "high", a), m(range_mf, "mid", r)),
                     np.fmin(m(density_mf, "high", d), m(iteration_mf, "low", it))), ale_mf["mid"]),

            (np.fmin(np.fmin(m(anchor_mf, "low", a), m(range_mf, "low", r)),
                     np.fmin(m(density_mf, "low", d), m(iteration_mf, "high", it))), ale_mf["high"]),

            (np.fmin(np.fmin(m(anchor_mf, "mid", a), m(range_mf, "high", r)),
                     np.fmin(m(density_mf, "low", d), m(iteration_mf, "mid", it))), ale_mf["mid"]),

            (np.fmin(np.fmin(m(anchor_mf, "low", a), m(range_mf, "mid", r)),
                     np.fmin(m(density_mf, "high", d), m(iteration_mf, "high", it))), ale_mf["mid"]),

            (np.fmin(np.fmin(m(anchor_mf, "mid", a), m(range_mf, "low", r)),
                     np.fmin(m(density_mf, "mid", d), m(iteration_mf, "high", it))), ale_mf["mid"]),

            (np.fmin(np.fmin(m(anchor_mf, "high", a), m(range_mf, "high", r)),
                     np.fmin(m(density_mf, "high", d), m(iteration_mf, "high", it))), ale_mf["low"]),

            (np.fmin(np.fmin(m(anchor_mf, "low", a), m(range_mf, "high", r)),
                     np.fmin(m(density_mf, "mid", d), m(iteration_mf, "low", it))), ale_mf["mid"]),

            (np.fmin(np.fmin(m(anchor_mf, "mid", a), m(range_mf, "mid", r)),
                     np.fmin(m(density_mf, "mid", d), m(iteration_mf, "mid", it))), ale_mf["mid"]),

            (np.fmin(np.fmin(m(anchor_mf, "high", a), m(range_mf, "low", r)),
                     np.fmin(m(density_mf, "low", d), m(iteration_mf, "mid", it))), ale_mf["high"]),

            (np.fmin(np.fmin(m(anchor_mf, "mid", a), m(range_mf, "mid", r)),
                     np.fmin(m(density_mf, "high", d), m(iteration_mf, "low", it))), ale_mf["low"]),
        ])

        combined_activation = np.zeros_like(ale_range)
        for strength, output_mf in rules:
            combined_activation = np.fmax(combined_activation, strength * output_mf)

        if combined_activation.max() > 0:
            if defuzz_method == 'centroid':
                ale_defuzz = fuzz.defuzz(ale_range, combined_activation, 'centroid')
            elif defuzz_method == 'weighted':
                weights = combined_activation
                values = ale_range
                ale_defuzz = np.sum(weights * values) / np.sum(weights)
            else:
                ale_defuzz = 0.5
        else:
            ale_defuzz = 0.5

        predictions.append(ale_defuzz)

    y_pred = np.array(predictions) * (y_max - y_min) + y_min
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    print(f"[{label}] MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    return label, mae, rmse



# Kombinasyonları çalıştır
results = []

tri = create_tri_membership(universe)
gauss = create_gauss_membership(universe)

# 4 kombinasyon
results.append(run_fuzzy_inference(X_normalized, tri, tri, tri, tri, tri, 'centroid', 'Tri + Centroid'))
results.append(run_fuzzy_inference(X_normalized, tri, tri, tri, tri, tri, 'weighted', 'Tri + Weighted'))
results.append(run_fuzzy_inference(X_normalized, gauss, gauss, gauss, gauss, gauss, 'centroid', 'Gauss + Centroid'))
results.append(run_fuzzy_inference(X_normalized, gauss, gauss, gauss, gauss, gauss, 'weighted', 'Gauss + Weighted'))

# Grafiğe dök
labels = [r[0] for r in results]
maes = [r[1] for r in results]
rmses = [r[2] for r in results]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, maes, width, label='MAE')
plt.bar(x + width/2, rmses, width, label='RMSE')
plt.xticks(x, labels)
plt.ylabel("Hata Değeri")
plt.title("Farklı Kombinasyonların MAE ve RMSE Karşılaştırması")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("kombinasyon_karsilastirma.png", dpi=300)
plt.show()
