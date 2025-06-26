from pymongo import MongoClient
from datetime import datetime
import cv2, os, signal
from feature_extraction import extract_features, combine_features
from config import MONGODB_URI, DB_NAME, COLLECTION_NAME, DEFAULT_WEIGHTS
from bson import Binary
import base64

# Class xử lý timeout
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Xử lý ảnh quá thời gian cho phép")

def connect_mongodb():
    client = MongoClient(MONGODB_URI)
    return client[DB_NAME][COLLECTION_NAME]

# Phiên bản cho Windows
import threading
import time

def process_image_with_timeout(image, timeout=10):
    result = {"success": False, "data": None, "error": None}
    
    def target_function():
        try:
            features_dict = extract_features(image)
            features = combine_features(features_dict, DEFAULT_WEIGHTS)
            result["success"] = True
            result["data"] = (features_dict, features)
        except Exception as e:
            result["error"] = e
    
    thread = threading.Thread(target=target_function)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        # Timeout xảy ra
        return None, None, TimeoutError("Xử lý ảnh quá thời gian cho phép")
    
    if not result["success"]:
        return None, None, result["error"]
    
    return result["data"][0], result["data"][1], None

def insert_image_features(image_folder):
    col = connect_mongodb()
    imgs = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    data = []
    skipped = 0
    processed = 0

    for idx, fname in enumerate(imgs):
        path = os.path.join(image_folder, fname)
        print(f"[{idx+1}/{len(imgs)}] Đang xử lý: {path}")
        
        try:
            image = cv2.imread(path)
            if image is None:
                print(f"⚠️ Không thể đọc ảnh: {fname}")
                skipped += 1
                continue
            
            try:
                # Xử lý ảnh với timeout 10 giây
                features_dict, features, error = process_image_with_timeout(image, timeout=10)
                
                if error:
                    if isinstance(error, TimeoutError):
                        print(f"⏱️ Vượt quá thời gian xử lý (10s) cho ảnh: {fname}")
                    else:
                        print(f"❌ Lỗi khi trích xuất đặc trưng cho {fname}: {error}")
                    skipped += 1
                    continue
                
                # Chuyển đổi ảnh thành định dạng Base64
                _, buffer = cv2.imencode('.jpg', image)
                image_base64 = base64.b64encode(buffer).decode('utf-8')

                height, width = image.shape[:2]
                ctime = os.path.getctime(path)
                created_time = datetime.fromtimestamp(ctime).isoformat()
                doc = {
                    "filename": fname,
                    "filepath": path,
                    "dimensions": {"height": height, "width": width},
                    "features": features.tolist(),
                    "created_time": created_time,
                    "image_base64": image_base64  # Lưu ảnh dạng Base64
                }
                data.append(doc)
                processed += 1
                
            except TimeoutError:
                print(f"⏱️ Vượt quá thời gian xử lý (10s) cho ảnh: {fname}")
                skipped += 1
                continue
            except Exception as e:
                print(f"❌ Lỗi khi trích xuất đặc trưng cho {fname}: {e}")
                skipped += 1
                continue
                
        except Exception as e:
            print(f"❌ Lỗi không xác định với {fname}: {e}")
            skipped += 1
            continue

    if data:
        col.insert_many(data)
        
    print(f"✅ Đã xử lý thành công: {processed} ảnh")
    print(f"⚠️ Đã bỏ qua: {skipped} ảnh")
    return True


def store_image_features(image_folder, rebuild=False):
    col = connect_mongodb()
    if rebuild or col.count_documents({}) == 0:
        if rebuild:
            col.delete_many({})

        imgs = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        data = []
        skipped = 0
        processed = 0

        for idx, fname in enumerate(imgs):
            path = os.path.join(image_folder, fname)
            print(f"[{idx+1}/{len(imgs)}] Đang xử lý: {path}")
            
            try:
                image = cv2.imread(path)
                if image is None:
                    print(f"⚠️ Không thể đọc ảnh: {fname}")
                    skipped += 1
                    continue
                
                try:
                    # Xử lý ảnh với timeout 10 giây
                    features_dict, features, error = process_image_with_timeout(image, timeout=10)
                    
                    if error:
                        if isinstance(error, TimeoutError):
                            print(f"⏱️ Vượt quá thời gian xử lý (10s) cho ảnh: {fname}")
                        else:
                            print(f"❌ Lỗi khi trích xuất đặc trưng cho {fname}: {error}")
                        skipped += 1
                        continue
                    
                    # Chuyển đổi ảnh thành định dạng Base64
                    _, buffer = cv2.imencode('.jpg', image)
                    image_base64 = base64.b64encode(buffer).decode('utf-8')

                    height, width = image.shape[:2]
                    ctime = os.path.getctime(path)
                    created_time = datetime.fromtimestamp(ctime).isoformat()
                    doc = {
                        "filename": fname,
                        "filepath": path,
                        "dimensions": {"height": height, "width": width},
                        "features": features.tolist(),
                        "created_time": created_time,
                        "image_base64": image_base64  # Lưu ảnh dạng Base64
                    }
                    data.append(doc)
                    processed += 1
                    
                except TimeoutError:
                    print(f"⏱️ Vượt quá thời gian xử lý (10s) cho ảnh: {fname}")
                    skipped += 1
                    continue
                except Exception as e:
                    print(f"❌ Lỗi khi trích xuất đặc trưng cho {fname}: {e}")
                    skipped += 1
                    continue
                    
            except Exception as e:
                print(f"❌ Lỗi không xác định với {fname}: {e}")
                skipped += 1
                continue

        if data:
            col.insert_many(data)
            
        print(f"✅ Đã xử lý thành công: {processed} ảnh")
        print(f"⚠️ Đã bỏ qua: {skipped} ảnh")
        return True
    else:
        print(f"Đã có sẵn {col.count_documents({})} bản ghi. Dùng rebuild=True nếu muốn tạo lại.")
        return False