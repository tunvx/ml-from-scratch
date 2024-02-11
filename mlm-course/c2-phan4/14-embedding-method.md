# Phương pháp nhúng

## 1. Lựa chọn theo mô hình hồi quy

Trong mô hình hồi quy, các hệ số tỷ lệ thuận với tác động của đặc trưng đó tới kết quả cuối cùng. Chúng ta có thể thấy rằng hệ số biểu thị ảnh hưởng hoặc mức độ quan trọng của biến cụ thể đó với kết quả, nhưng độ quan trọng này cần được thực hiện một cách cẩn trọng, bởi vì các hệ số này phụ thuộc vào nhiều thứ khác nhau.

 

Để so sánh đặc trưng bằng cách xem xét các hệ số, chúng ta cần đảm bảo nén các đặc trưng nằm trong khoảng từ 0 đến 1 hoặc -1 đến 1 hoặc chuẩn tắc hóa các đặc trưng.

- **Video: [Hệ số hồi quy](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/9341782#overview)**
- **Video: [Lựa chọn theo hệ số hồi quy tuyến tính](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/9533576#overview)**

Điều chuẩn gồm việc thêm một lượng phạt với các tham số khác nhau của mô hình để giảm tự do của mô hình, do đó mô hình sẽ ít có khả năng khớp với nhiễu của dữ liệu huấn luyện và cải thiện khả năng tổng quát hóa của thuật toán học máy. 

Với các mô hình tuyến tính, nhìn chung có 3 loại điều chuẩn:

- - Điều chuẩn L1 (Lasso)
  - điều chuẩn L2 (Ridge)
  - Điều chuẩn L1/L2 (Elastic net)

- **Video: [Điều chuẩn](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/9341774#overview)**

Điều chuẩn Ridge đồng loạt giảm tất cả các hệ số, chỉ tới 0 khi lượng phạt rất cao. Ngược lại, với Lasso, các hệ số của các đặc trưng khác nhau lần lượt thu về thành 0. Theo cách này, chúng ta có thể lựa chọn đặc trưng trong khi khớp với thuật toán học máy.

- **Video: [Lasso](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/9523068#overview)**

## 2. Lựa chọn theo các phương pháp lai hóa

Có những phương pháp khác không nằm trong bất kỳ hạng mục nào, nhưng có các đặc điểm của cả hai hạng mục nên được gọi là phương pháp lai hóa (Hybrid method).

- **Video: [Giới thiệu về các phương pháp lai hóa](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22610066#overview)**

Phương pháp phổ biến để lựa chọn đặc trưng gồm xáo trộn ngẫu nhiên các giá trị của một biến cụ thể và xác định hoán vị đó ảnh hưởng thế nào đến phép đo chất lượng của thuật toán học máy. Nói cách khác, ý tưởng là hoán vị các giá trị của từng đặc trưng tại thời điểm đó và đo lường mức độ hoán vị (hoặc xáo trộn các giá trị của nó) làm giảm độ chính xác hay roc_auc hoặc mse của mô hình học máy (hoặc bất kỳ phép đo chất lượng nào khác!). Nếu các biến quan trọng thì hoán vị ngẫu nhiên các giá trị sẽ giảm đáng kể bất kỳ phép đo nào trong số này. Ngược lại, hoán vị hoặc xáo trộn các giá trị sẽ ít hoặc không ảnh hưởng đến phép đo chất lượng mô hình mà chúng ta đang đánh giá.

Quy trình sẽ như sau:

1. 1. Xây dựng mô hình học máy và lưu trữ phép đo chất lượng.
   2. Xáo trộn 1 đặc trưng và đưa ra dự đoán mới sử dụng mô hình trước đó.
   3. Xác định chất lượng của dự đoán này.
   4. Xác định thay đổi về chất lượng của dự đoán với các đặc trưng đã xáo trộn với đặc trưng ban đầu.
   5. Lặp lại cho từng đặc trưng.



- **Video: [Xáo trộn đặc trưng](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22609732#overview)**

Phương pháp loại bỏ đặc trưng bằng đệ quy gồm các bước sau:

1. 1. Xếp hạng các đặc trưng theo mức độ quan trọng của chúng từ thuật toán học máy: có thể là mức độ quan trọng của cây hoặc các hệ số thu được từ mô hình tuyến tính.
   2. Loại bỏ đặc trưng ít quan trọng nhất và xây dựng thuật toán học máy bằng với các đặc trưng còn lại.
   3. Tính toán phép đo chất lượng được lựa chọn: roc-auc, mse, rmse, accuracy,...
   4. Nếu phép đo giảm nhiều hơn ngưỡng được thiết lập tùy ý thì đặc trưng đó quan trọng và cần được giữ lại; nếu không, chúng ta có thể loại bỏ đặc trưng đó.
   5. Lặp lại các bước 2-4 cho đến khi tất cả các đặc trưng đã được đánh giá.

- **Video: [Loại bỏ đặc trưng bằng đệ quy](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22609730#overview)**

Phương pháp thêm đặc trưng bằng đệ quy gồm các bước sau:

1. Xếp hạng các đặc trưng theo mức độ quan trọng của chúng từ thuật toán học máy: có thể là mức độ quan trọng của cây hoặc các hệ số thu được từ mô hình tuyến tính.
2. Xây dựng mô hình học máy chỉ với 1 đặc trưng, đặc trưng quan trọng nhất và tính toán phép đo chất lượng.
3. Thêm một đặc trưng - đặc trưng quan trọng nhất trong nhóm các đặc trưng còn lại và xây dựng thuật toán học máy sử dụng đặc trưng đã thêm và bất kỳ đặc trưng nào từ các vòng trước đó.
4. Tính toán phép đo chất lượng đã chọn chọn: roc-auc, mse, rmse, accuracy,...
5. Nếu phép đo tăng nhiều hơn ngưỡng được thiết lập tùy ý thì đặc trưng đó quan trọng và cần được giữ lại; nếu không, chúng ta có thể loại bỏ đặc trưng đó.
6. Lặp lại các bước 2-5 cho đến khi tất cả các đặc trưng đã được đánh giá.

- **Video: [Thêm đặc trưng bằng đệ quy](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22609724#overview)**