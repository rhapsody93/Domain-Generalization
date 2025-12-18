import os
import shutil
import random
import json
import glob
from pathlib import Path
import yaml
import numpy as np
from ultralytics import YOLO

# =========================
# 기본 경로
# =========================
BASE_DIR = Path("/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/related_codes/domain_generalization/leave_one_weather_out")
SD_DIR = Path("/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/source_domain_weather/MERGED_SD_BB")
TD_DIR = Path("/home/jhkim/anaconda3/envs/project_ms/custom_datasets/Object_Detection/target_domain_weather/MERGED_TD_BB")

# AI-Hub JSON Metadata
AIHUB_JSON_BASE = "/home/jhkim/anaconda3/envs/project_ms/original_datasets/target_domain_weather/adverse_weather_datasets/AI-Hub/autonomous_driving_adverse_weather/open_source_data/Training/02.labeling_data/TL/2D"

# 클래스 구성 (YOLO names)
SD_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'traffic light', 'bus', 'train', 'truck', 'traffic sign']
TD_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'traffic light', 'bus', 'train', 'truck', 'traffic sign']

# 날씨 키워드 (Leave-One-Weather-Out 대상)
WEATHER_KEYWORDS = ["fog", "rain", "snow", "sand", "night", "cloudy"]


# =========================
# WEATHER MAPPING (fog, rain, snow, sand, night, cloudy)
# =========================
def _normalize_weather_key(w):
    if w is None:
        return "unknown"
    s = str(w).strip().lower()
    if any(k in s for k in ["fog", "foggy", "mist", "haze", "안개"]):
        return "fog"
    if any(k in s for k in ["rain", "비"]):
        return "rain"
    if any(k in s for k in ["snow", "눈"]):
        return "snow"
    if any(k in s for k in ["sand", "dust", "모래"]):
        return "sand"
    if any(k in s for k in ["night", "야간"]):
        return "night"
    if any(k in s for k in ["흐림", "cloud", "cloudy", "overcast"]):
        return "cloudy"
    if "맑" in s or "clear" in s:
        return "clear"
    return "unknown"


def extract_weather_from_aih_json(img_name):
    """
    AI-Hub 원본 JSON에서 weather를 추출하여 표준화된 weather로 반환.
    img_name 예: '08_084144_221012_08.jpg'
    """
    parts = img_name.split("_")
    if len(parts) < 3:
        return "unknown"
    scene_id = "_".join(parts[:3])
    search_path = os.path.join(AIHUB_JSON_BASE, scene_id, "sensor_raw_data/camera")
    if not os.path.exists(search_path):
        return "unknown"
    for jf in glob.glob(os.path.join(search_path, "*.json")):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                weather_str = data.get("Environment_meta", {}).get("weather", "")
                return _normalize_weather_key(weather_str)
        except Exception:
            continue
    return "unknown"


def get_weather_label(img_path: Path):
    """
    MERGED_TD_BB / MERGED_SD_BB 공통 weather 라벨링 함수.
    - ACDC/DAWN : 파일명 기반
    - AI-Hub     : JSON 기반
    """
    fname = os.path.basename(img_path).lower()

    # ACDC/DAWN: 파일명 기반
    if any(k in fname for k in ["fog_", "foggy", "mist", "haze"]):
        return "fog"
    if any(k in fname for k in ["rain_", "rain", "rain_storm"]):
        return "rain"
    if any(k in fname for k in ["snow_", "snow", "snow_storm"]):
        return "snow"
    if any(k in fname for k in ["sand", "dust", "dusttornado"]):
        return "sand"
    if any(k in fname for k in ["night_", "night"]):
        return "night"

    # AI-Hub: JSON 기반
    w = extract_weather_from_aih_json(os.path.basename(img_path))
    if w == "clear":
        # clear(맑음)은 DG 실험에서는 'unknown'으로 버림
        return "unknown"
    return w


# =========================
# 평가 메트릭 계산 함수 (DTDR / RI)
# =========================
def domain_transfer_degradation_rate(source_score: float, target_score: float) -> float:
    """
    Domain Transfer Degradation Rate (성능 하락률 %)
    """
    return (source_score - target_score) / max(source_score, 1e-6) * 100.0


def robustness_index(scores):
    """
    Robustness Index (분산 기반 강건성 지표)
    """
    arr = np.array(scores, dtype=float)
    if arr.size == 0:
        return 0.0
    mean, std = arr.mean(), arr.std()
    if mean == 0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - (std / mean))))


# =========================
# 이미지-라벨 유효성 검사
# =========================
def valid_img_label_pairs(img_paths):
    """
    원본 MERGED_SD_BB / MERGED_TD_BB 구조에서
    - images/.../xxx.jpg 에 대응하는 labels/.../xxx.txt 가 있는 경우만 사용
    """
    valid_pairs = []
    for img in img_paths:
        lbl = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
        if lbl.exists():
            valid_pairs.append(img)
    return valid_pairs


# =========================
# Strict balanced sampling 유틸
# =========================
def strict_sample_weather(img_paths, target_count):
    """
    weather별 target_count(Train: 250, Val: 49)를 맞추기 위한 샘플러.
    - label 없는 이미지 제거
    - 부족하면 oversampling으로 채움
    """
    img_paths = valid_img_label_pairs(img_paths)
    if len(img_paths) == 0:
        return []
    if len(img_paths) >= target_count:
        return random.sample(img_paths, target_count)
    # 부족하면 중복 허용 (oversampling)
    return random.choices(img_paths, k=target_count)


def copy_with_label(src_img: Path, dst_img_dir: Path, dst_lbl_dir: Path):
    """
    원본 MERGED_* 구조에서 이미지와 라벨을 함께 복사.
    src_img: 원본 images/.../xxx.jpg
    dst_img_dir: 새로운 dataset_root/images/{split}
    dst_lbl_dir: 새로운 dataset_root/labels/{split}
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 복사
    shutil.copy(src_img, dst_img_dir / src_img.name)

    # 라벨 복사
    src_lbl = Path(str(src_img).replace("/images/", "/labels/")).with_suffix(".txt")
    if src_lbl.exists():
        shutil.copy(src_lbl, dst_lbl_dir / src_lbl.name)


# =========================
# Strict balanced Train/Val 생성
# =========================
def create_balanced_splits(dataset_root: Path, excluded_weather: str,
                           train_per_weather: int = 250, val_per_weather: int = 49):
    """
    dataset_root/ 아래에 YOLO 표준 구조로 strict balanced train/val 생성:
    - images/train, labels/train
    - images/val,   labels/val
    구성 규칙:
      * Train: clear(SD) + (non-target) 5개 날씨(TD) 각 250장
      * Val:   clear(SD) + (non-target) 5개 날씨(TD) 각 49장
    """

    # 기존 디렉토리 정리
    for split in ["train", "val"]:
        for kind in ["images", "labels"]:
            d = dataset_root / kind / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    # 1) 원본 파일 리스트
    sd_train_imgs = list((SD_DIR / "images/train").glob("*.jpg"))
    sd_val_imgs   = list((SD_DIR / "images/val").glob("*.jpg"))
    td_train_imgs = list((TD_DIR / "images/train").glob("*.jpg"))
    td_val_imgs   = list((TD_DIR / "images/val").glob("*.jpg"))

    # 2) weather 그룹 (clear는 SD 전용)
    weathers = ["clear", "fog", "rain", "snow", "sand", "night", "cloudy"]

    train_groups = {w: [] for w in weathers}
    val_groups   = {w: [] for w in weathers}

    # SD → clear (train/val)
    for img in sd_train_imgs:
        train_groups["clear"].append(img)
    for img in sd_val_imgs:
        val_groups["clear"].append(img)

    # TD → weather (train/val)
    for img in td_train_imgs:
        w = get_weather_label(img)
        if w in ["unknown", "clear"]:
            continue
        if w == excluded_weather:
            continue
        if w in train_groups:
            train_groups[w].append(img)

    for img in td_val_imgs:
        w = get_weather_label(img)
        if w in ["unknown", "clear"]:
            continue
        if w == excluded_weather:
            continue
        if w in val_groups:
            val_groups[w].append(img)

    # 3) strict balanced sampling & 복사
    # Train
    train_used = {}
    for w in weathers:
        if w == excluded_weather:
            continue
        selected = strict_sample_weather(train_groups[w], train_per_weather)
        train_used[w] = len(selected)
        for img in selected:
            copy_with_label(img, dataset_root / "images/train", dataset_root / "labels/train")

    # Val
    val_used = {}
    for w in weathers:
        if w == excluded_weather:
            continue
        selected = strict_sample_weather(val_groups[w], val_per_weather)
        val_used[w] = len(selected)
        for img in selected:
            copy_with_label(img, dataset_root / "images/val", dataset_root / "labels/val")

    print(f"[INFO][TRAIN] strict balanced counts for target '{excluded_weather}': {train_used}")
    print(f"[INFO][VAL]   strict balanced counts for target '{excluded_weather}': {val_used}")

    # sanity check
    n_train_imgs = len(list((dataset_root / "images/train").glob("*.jpg")))
    n_train_lbls = len(list((dataset_root / "labels/train").glob("*.txt")))
    n_val_imgs   = len(list((dataset_root / "images/val").glob("*.jpg")))
    n_val_lbls   = len(list((dataset_root / "labels/val").glob("*.txt")))
    print(f"[INFO] Train images: {n_train_imgs}, Train labels: {n_train_lbls}")
    print(f"[INFO] Val   images: {n_val_imgs}, Val   labels: {n_val_lbls}")


# =========================
# Target Weather Test set 생성 (옵션 A: 원본 test 전체 사용)
# =========================
def create_target_testset(dataset_root: Path, target_weather: str):
    """
    MERGED_TD_BB test split에서 target_weather 이미지만 모아서
    dataset_root/images/test, labels/test 에 구성.
    (oversampling 없음, 원본 test 분포 그대로 사용)
    """
    img_dir = dataset_root / "images/test"
    lbl_dir = dataset_root / "labels/test"

    if img_dir.exists():
        shutil.rmtree(img_dir)
    if lbl_dir.exists():
        shutil.rmtree(lbl_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    td_test_imgs = list((TD_DIR / "images/test").glob("*.jpg"))
    count = 0
    for img in td_test_imgs:
        w = get_weather_label(img)
        if w != target_weather:
            continue
        copy_with_label(img, img_dir, lbl_dir)
        count += 1

    print(f"[INFO][TEST] Target weather '{target_weather}' test images: {count}")


# =========================
# YAML 생성
# =========================
def create_yaml(dataset_root: Path, yaml_path: Path, names):
    """
    YOLO standard dataset structure:
      path: dataset_root
      train: images/train
      val  : images/val
      test : images/test
    """
    yaml_dict = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Created YAML: {yaml_path}")
    return yaml_path


# =========================
# YOLO 평가 메트릭 함수
# =========================
def evaluate_yolo_metrics(model_path, data_yaml, split="val"):
    """
    YOLO .val() 결과에서 box mAP, precision, recall, F1 추출
    """
    model = YOLO(model_path)
    res = model.val(data=data_yaml, split=split, imgsz=640)

    f1_attr = getattr(res.box, 'f1', 0.0)
    if isinstance(f1_attr, (list, np.ndarray)):
        f1_value = float(np.mean(f1_attr))
    else:
        f1_value = float(f1_attr)

    metrics = {
        "mAP50": float(res.box.map50),
        "mAP50-95": float(res.box.map),
        "Precision": float(res.box.mp),
        "Recall": float(res.box.mr),
        "F1": f1_value
    }
    return metrics


# =========================
# 학습-검증-테스트 파이프라인
# =========================
def run_pipeline(weather):
    """
    Leave-One-Weather-Out (strict balanced, 옵션 A):
      (1) Train: clear + (6개 중 target 제외 5개) 각 250장
      (2) Val  : clear + (6개 중 target 제외 5개) 각 49장
      (3) Test : MERGED_TD_BB test split에서 target weather 전체 (oversampling 없음)
    """
    print(f"\n========== Leave-One-Weather-Out (Strict Balanced, Option A): {weather.upper()} ==========")

    dataset_root = BASE_DIR / f"strict_balanced_{weather}"
    dataset_root.mkdir(parents=True, exist_ok=True)

    # 1. Train/Val strict balanced
    create_balanced_splits(dataset_root,
                           excluded_weather=weather,
                           train_per_weather=250,
                           val_per_weather=49)

    # 2. Test: target weather만, 원본 test 전체
    create_target_testset(dataset_root, target_weather=weather)

    # 3. YAML 생성
    yaml_path = BASE_DIR / f"{weather}_strict_balanced.yaml"
    create_yaml(dataset_root, yaml_path, SD_CLASSES)

    # 4. YOLOv8l 학습
    model = YOLO("yolov8l.pt")
    work_dir = BASE_DIR / f"runs_{weather}"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Training YOLOv8l (strict balanced) excluding target weather '{weather}'...")
    model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=640,
        batch=4,
        workers=2,
        project=str(work_dir),
        name="train"
    )

    best_weight = work_dir / "train" / "weights" / "best.pt"

    # 5. 검증 (source-like, non-target weathers + clear)
    val_metrics = evaluate_yolo_metrics(str(best_weight), str(yaml_path), split="val")
    print(f"[INFO] Validation metrics (target '{weather}' excluded, strict balanced): {val_metrics}")

    # 6. 테스트 (target weather only, 원본 test 전체)
    test_metrics = evaluate_yolo_metrics(str(best_weight), str(yaml_path), split="test")
    print(f"[INFO] Test metrics (target weather '{weather}' only, full test set): {test_metrics}")

    # 7. DTDR, RI 계산
    dtdr = domain_transfer_degradation_rate(val_metrics["mAP50"], test_metrics["mAP50"])
    ri = robustness_index([val_metrics["mAP50"], test_metrics["mAP50"]])
    print(f"[DTDR] {dtdr:.2f}% | [RI] {ri:.4f}")

    # 8. JSON 저장
    summary = {
        "Weather": weather,
        "Validation": val_metrics,
        "Test": test_metrics,
        "DTDR(%)": dtdr,
        "Robustness Index": ri
    }
    with open(work_dir / f"summary_{weather}.json", "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[INFO] Saved summary JSON: {work_dir / f'summary_{weather}.json'}")

    # 9. 테스트셋 오버레이 저장
    trained_model = YOLO(str(best_weight))
    print(f"[INFO] Saving overlay predictions for target weather '{weather}' test set...")
    trained_model.predict(
        source=str(dataset_root / "images/test"),
        imgsz=640,
        conf=0.25,
        project=str(work_dir),
        name="test_overlay",
        save=True
    )


# =========================
# 메인 실행
# =========================
if __name__ == "__main__":
    random.seed(42)
    for weather in WEATHER_KEYWORDS:
        run_pipeline(weather)
