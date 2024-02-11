# Phương pháp lọc



## 1. Lọc các đặc trưng trùng lặp

- Video: [Các đặc trưng không đổi, gần như không đổi và trùng lặp](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/9341710#overview)

## 2. Lọc đặc trưng theo tương quan

- Video: [Tương quan](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22474122#overview)
- Video: [Lựa chọn đặc trưng theo tương quan](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/9341720#overview)
- Video: [Quy trình lựa chọn đặc trưng theo tương quan](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22475392#overview)

## 3. Lọc đặc trưng bằng các phép đo thống kê

- **Video**: [Phương pháp thống kê](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/9341732#overview)

- **Video**: [Thông tin tương hỗ](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22494300#overview)
- **Video**: [Chi-square](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22495182#overview)

- **Video**: [Anova](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22495194#overview)

## 4. Lọc đặc trưng bằng các phép đo khác

- **Video**: [Phương pháp lọc với các phép đo khác](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22525918#overview)

Phương pháp đo chất lượng cho mô hình đơn biến hoạt động theo quy trình như sau:

- - Xây dựng mô hình trên mỗi đặc trưng để dự đoán mục tiêu.
  - Đưa ra dự đoán sử dụng mô hình được tạo ra từ đặc trưng đã đề cập.
  - Đo lường chất lượng của dự đoán đó, có thể là roc-auc (bài toàn phân loại), mse (bài toán hồi quy).
  - Xếp hạng các đặc trưng theo phép đo (roc-auc hoặc mse).
  - Chọn ra các đặc trưng có xếp hạng cao nhất.

- **Video**: [Phương pháp đo chất lượng cho mô hình đơn biến](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22525920#overview)

Lựa chọn đặc trưng bằng mã hóa mục tiêu được sử dụng trong cuộc thi KDD 2009 và nó liên quan đến việc mã hóa các biến cả dạng số và phân loại với biểu diễn của mục tiêu, sau đó sử dụng phép biểu diễn này làm dự đoán.