import pandas as pd
from typing import Dict, List, Tuple

def load_rules(file_path: str = './rules.csv') -> List[Dict]:
    df = pd.read_csv(file_path)
    if 'conclusion' not in df.columns or 'cf' not in df.columns:
        raise ValueError("File harus berisi kolom 'conclusion' dan 'cf'.")

    condition_cols = [c for c in df.columns if c not in ('conclusion', 'cf')]
    rules: List[Dict] = []
    for record in df.to_dict(orient='records'):
        conditions = {
            col: str(val)
            for col, val in record.items()
            if col in condition_cols and pd.notna(val) and val != ''
        }
        rules.append({
            'conditions': conditions,
            'conclusion': str(record['conclusion']),
            'cf': float(record['cf'])
        })
    return rules

RULES = load_rules()

# rumus
# jika sama: cf1 + cf2 * (1-abs(cf1))
# jika tidak sama: (cf1+cf2)/(1-min(abs(cf1),abs(cf2)))
def combine_cf(cf1: float, cf2: float) -> float:
    if cf1 * cf2 >= 0:
        return cf1 + cf2 * (1 - abs(cf1))
    return (cf1 + cf2) / (1 - min(abs(cf1), abs(cf2)))

def infer_cf(user_profile: Dict[str, str]) -> Dict[str, float]:
    cfs: Dict[str, float] = {}
    for rule in RULES:
        if all(user_profile.get(attr) == val for attr, val in rule['conditions'].items()):
            key = rule['conclusion']
            cfs[key] = combine_cf(cfs.get(key, 0.0), rule['cf'])
    return cfs

def recommend_style(user_profile: Dict[str, str]) -> Tuple[str, float]:
    cfs = infer_cf(user_profile)
    if not cfs:
        return None, 0.0
    style, cf_val = max(cfs.items(), key=lambda x: x[1])
    return style, cf_val

if __name__ == '__main__':
    print("Silakan masukkan profil Anda dengan memilih dari opsi yang tersedia.\n")

    profil = {
        'Gender': input("Gender (contoh: Male / Female): "),
        'HeightCategory': input("Height Category (contoh: Short / Medium / Tall): "),
        'BodyType': input("Body Type (contoh: Slim / Average / Heavy): "),
        'SkinTone': input("Skin Tone (contoh: Fair / Medium / Dark): "),
        'FaceShape': input("Face Shape (contoh: Oval / Round / Square / Heart / etc): "),
        'HairLength': input("Hair Length (contoh: Short / Medium / Long): ")
    }

    gaya, nilai_cf = recommend_style(profil)
    if gaya:
        print(f"\nRekomendasi gaya: {gaya} (CF={nilai_cf:.2f})")
    else:
        print("\nTidak ada rekomendasi yang cocok.")

