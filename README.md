import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ==========================================
# محاسبه Isd و Isq
# ==========================================
def compute_isd_isq(Amax, Bmax, Cmax):
    Isd = np.sqrt(2/3)*Amax - np.sqrt(1/6)*Bmax - np.sqrt(1/6)*Cmax
    Isq = np.sqrt(1/2)*(Bmax - Cmax)
    return Isd, Isq


# ==========================================
# تحلیل سلامت + یادگیری ماشین
# ==========================================
def bearing_health_analysis_with_ml(file_path):

    # ---------- خواندن فایل ----------
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError("File must be Excel or CSV")

    # ---------- بررسی ستون‌ها ----------
    for col in ["Amax2", "Bmax2", "Cmax2"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing")

    if "Motor_ID" not in df.columns:
        df["Motor_ID"] = np.arange(1, len(df)+1)

    # ---------- Isd , Isq , r ----------
    Isd_list, Isq_list, r_list = [], [], []

    for _, row in df.iterrows():
        Isd, Isq = compute_isd_isq(row["Amax2"], row["Bmax2"], row["Cmax2"])
        r = np.sqrt(Isd**2 + Isq**2)

        Isd_list.append(Isd)
        Isq_list.append(Isq)
        r_list.append(r)

    df["Isd"] = Isd_list
    df["Isq"] = Isq_list
    df["r"]   = r_list

    # ---------- r_fit ----------
    r_fit = df["r"].mean()

    # ---------- δ ----------
    df["delta_raw"] = np.abs(df["r"] - r_fit) / r_fit
    df["delta_percent"] = df["delta_raw"] * 100

    # ---------- شاخص سلامت ----------
    df["HI"] = np.exp(-df["delta_percent"])

    # ---------- RUL فیزیکی ----------
    RUL_MAX = 600  # روز
    df["RUL_physical"] = RUL_MAX * df["HI"]

    # ==========================================
    # یادگیری ماشین
    # ==========================================
    X = df[["Isd", "Isq", "r", "delta_percent"]].values
    y = df["RUL_physical"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    df["RUL_ML"] = model.predict(X_scaled)

    # ---------- وضعیت ----------
    df["Status"] = df["delta_percent"].apply(
        lambda x: "Healthy" if x < 5 else "Faulty"
    )

    return df, r_fit, model


# ==========================================
# اجرای برنامه
# ==========================================
file_path = "csv.xlsx"

result_df, r_fit, ml_model = bearing_health_analysis_with_ml(file_path)

print("Reference r_fit =", r_fit)
print(result_df)

# ذخیره خروجی
result_df.to_excel("bearing_health_ml_results.xlsx", index=False)
print("\nخروجی در فایل 'bearing_health_ml_results.xlsx' ذخیره شد.")
