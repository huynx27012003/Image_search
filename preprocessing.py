# preprocessing.py
import pickle
from sklearn.decomposition import PCA
import numpy as np
from config import REDUCED_DIM, REDUCER_PATH

class FeatureReducer:
    """
    Áp dụng PCA để giảm chiều vector đặc trưng đã kết hợp.
    Khởi tạo với số lượng thành phần mong muốn lấy từ config.REDUCED_DIM.
    Cho phép save/load object PCA để tái sử dụng cho truy vấn.
    """
    def __init__(self, n_components=REDUCED_DIM):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.fitted = False

    def fit(self, feature_matrix: np.ndarray):
        """
        Học PCA trên ma trận (n_samples, n_features).
        Trả về self để chain.
        """
        self.pca.fit(feature_matrix)
        self.fitted = True
        return self

    def transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Áp dụng PCA cho feature_matrix. Nếu chưa fit, sẽ load từ REDUCER_PATH.
        """
        if not self.fitted:
            self.load()
        return self.pca.transform(feature_matrix)

    def fit_transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Kết hợp fit và transform trên cùng dữ liệu.
        """
        self.fit(feature_matrix)
        return self.pca.transform(feature_matrix)

    def save(self, path=REDUCER_PATH):
        """
        Lưu object PCA đã train vào file (pickle).
        """
        with open(path, "wb") as f:
            pickle.dump(self.pca, f)

    def load(self, path=REDUCER_PATH):
        """
        Load object PCA từ file và đánh dấu là đã fit.
        """
        with open(path, "rb") as f:
            self.pca = pickle.load(f)
        self.fitted = True
        return self