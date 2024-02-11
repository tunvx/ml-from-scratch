# Decision Tree

**1. Giới thiệu về cây quyết định (DT)**

Hãy tìm hiểu một loại classifier khác: cây quyết định (DT) vô cùng hữu ích trong thực tế, đặc biệt là khi nó kết hợp với boosting (sẽ tìm hiểu ở bài sau).

- **Video:** [Dự đoán khả năng chi trả nợ với DT](https://www.coursera.org/learn/ml-classification/lecture/ZtvkP/predicting-loan-defaults-with-decision-trees)
- **Video:** [Ý tưởng đằng sau DT](https://www.coursera.org/learn/ml-classification/lecture/F8kuT/intuition-behind-decision-trees)
- **Video:** [Học tập từ dữ liệu với thuật toán DT](https://www.coursera.org/learn/ml-classification/lecture/xilmJ/task-of-learning-decision-trees-from-data)

**2. Huấn luyện DT**

Tiếp theo, chúng ta sẽ đi sâu vào một số chi tiết cốt lõi để triển khai. Chúng ta sẽ tìm hiểu ý tưởng về việc lựa chọn đặc trưng để phân chia, lặp lại ra sao và khi nào thì quyết định dừng.

- **Video:** [Thuật toán đệ quy tham lam](https://www.coursera.org/learn/ml-classification/lecture/Oor8r/recursive-greedy-algorithm)
- **Video:** [Huấn luyện ranh giới quyết định](https://www.coursera.org/learn/ml-classification/lecture/U7PcP/learning-a-decision-stump)
- **Video:** [Chọn đặc trưng tốt nhất để phân chia](https://www.coursera.org/learn/ml-classification/lecture/9RN9F/selecting-best-feature-to-split-on)
- **Video:** [Khi nào thì dừng đệ quy?](https://www.coursera.org/learn/ml-classification/lecture/fTlJU/when-to-stop-recursing)

**3. Sử dụng mô hình DT**

Chúng ta đã tìm hiểu DT từ dữ liệu. Hãy xem chúng ta có thể đưa ra những dự đoán gì từ nó.

- **Video:** [Dự đoán kết quả với DT](https://www.coursera.org/learn/ml-classification/lecture/HM4VD/making-predictions-with-decision-trees)
- **Video:** [Phân loại đa lớp với DT](https://www.coursera.org/learn/ml-classification/lecture/IVMdN/multiclass-classification-with-decision-trees)

**4. DT với biến liên tục**

Chúng ta xem xét một đặc trưng giá trị thực và thấy nó có các giá trị liên tục. Câu hỏi đặt ra là làm cách nào để xây dựng một DT với loại đầu vào này?

- **Video:** [Ngưỡng phân chia cho các đầu vào liên tục](https://www.coursera.org/learn/ml-classification/lecture/tn6M9/threshold-splits-for-continuous-inputs)
- **Video:** [(THAM KHẢO) Chọn ngưỡng tốt nhất để phân chia](https://www.coursera.org/learn/ml-classification/lecture/sKrGp/optional-picking-the-best-threshold-to-split-on)
- **Video:** [Trực quan hóa ranh giới quyết định](https://www.coursera.org/learn/ml-classification/lecture/kyi11/visualizing-decision-boundaries)

**5. Quá khớp trong DT**

DT rất dễ bị overfitting. Ở phần này, chúng ta sẽ tìm hiểu cách tránh overfitting trong trường hợp của DT.

- **Video:** [Nhắc lại về quá khớp](https://www.coursera.org/learn/ml-classification/lecture/czRmA/a-review-of-overfitting)
- **Video:** [Quá khớp trong DT](https://www.coursera.org/learn/ml-classification/lecture/XcPVL/overfitting-in-decision-trees)

Ý tưởng thú vị: tìm cây đơn giản hơn để giải thích dữ liệu đơn giản hơn là một khái niệm có từ lâu đời, gọi là lý thuyết dao cạo Occam.

- **Video:** [Lý thuyết dao cạo Occam: Huấn luyện các mô hình đơn giản hơn](https://www.coursera.org/learn/ml-classification/lecture/tUvBS/principle-of-occams-razor-learning-simpler-decision-trees)
- **Video:** [Dừng sớm (early stopping) trong DT](https://www.coursera.org/learn/ml-classification/lecture/gCuZ8/early-stopping-in-learning-decision-trees)
- **Video:** [Tóm tắt về quá khớp và điều chuẩn trong DT](https://www.coursera.org/learn/ml-classification/lecture/bRwHo/recap-of-overfitting-and-regularization-in-decision-trees)

**6. Kỹ thuật tỉa DT (tham khảo)**

Trong phần này, chúng ta sẽ tìm cách tốt nhất để ngăn chặn overfitting ở DT - sử dụng kỹ thuật cắt tỉa: tạo một cây lớn hơn cần thiết rồi cắt tỉa các phần ít quan trọng hơn.

- **Video:** [(THAM KHẢO) Động lực cắt tỉa](https://www.coursera.org/learn/ml-classification/lecture/9nMdb/optional-motivating-pruning)
- **Video:** [(THAM KHẢO) Cắt tỉa DT để tránh quá khớp](https://www.coursera.org/learn/ml-classification/lecture/qvf6v/optional-pruning-decision-trees-to-avoid-overfitting)
- **Video:** [(THAM KHẢO) Thuật toán cắt tỉa cây](https://www.coursera.org/learn/ml-classification/lecture/wmODB/optional-tree-pruning-algorithm)
- ***\*Tài liệu đọc tham khảo:\**** [Cây quyết định](https://drive.google.com/file/d/1-15xGNtdsD5xKvxL1PJeSSceseTiCj4r/view)



### Lab 9: Xác định các khoản vay an toàn với Decision Tree

- [Notebook](https://drive.google.com/drive/folders/10QMFY1b09grOOJk3RmgHAB8DwdUEph9E?usp=sharing)
- [Data](https://drive.google.com/file/d/14M3aE1z4WBKGh6Fo11EYrM7GPlL0E2o-/view)