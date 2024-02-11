# Oversampling

### 1. Trích xuất mẫu

Trong Random Oversampling, chúng ta trích xuất ngẫu nhiên các quan sát của lớp thiểu số cho đến khi đạt được tỷ lệ cân bằng nhất định. Đây là một kỹ thuật ngây thơ vì nó chỉ lấy mẫu ngẫu nhiên mà không đưa ra bất kỳ giả định nào về phân phối. Bằng cách trích xuất các mẫu một cách ngẫu nhiên, chúng ta thực ra đang sao chép các mẫu của lớp thiểu số để trong tập dữ liệu cuối cùng, chúng ta sẽ có nhiều quan sát giống hệt nhau. Điều này có thể khiến mô hình học máy overfit.

Để tạo ít mẫu trùng lặp hơn, chúng ta có thể giảm tỷ lệ cân bằng để trích xuất ít mẫu từ lớp thiểu số hơn. Chúng ta cần thử phương pháp này với các tỷ lệ cân bằng khác nhau và xem liệu nó có tạo thêm lợi thế về chất lượng mô hình không.

- **Video:** [Random Over-Sampling](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/23043534#overview)

### 2. Tạo mẫu

SMOTE (Synthetic Minority Over-sampling Technique) tạo ra các quan sát mới của lớp thiểu số bằng cách nội suy. Nội suy (Interpolation) là một loại ước tính mà chúng ta tạo các điểm dữ liệu mới trong phạm vi của các điểm dữ liệu đã biết.

SMOTE nội suy các mẫu mới theo hướng giữa mẫu ban đầu mà chúng ta lấy làm template và bất kỳ neighbour nào trong k neighbour của nó, với mẫu ban đầu và các neighbour này là kết quả của KNN chạy trên mẫu ban đầu (k thường là 5).

- **Video:** [SMOTE](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/23043540#overview)

MOTE-NC (SMOTE - Nominal Continuous) về cơ bản mở rộng chức năng của thuật toán SMOTE mà chúng ta đã mô tả trong video trước để nó có thể hoạt động với biến hạng mục. Đó là những đặc trưng có giá trị là hạng mục hoặc string thay vì số.

Làm thế nào để tính toán được khoảng cách giữa các biến hạng mục? Các giá trị giống nhau sẽ có khoảng cách bằng 0, các giá trị khác nhau chúng ta sẽ dùng trung vị của độ lệch chuẩn của các biến số còn lại.

Sau đó, làm cách nào để tạo các mẫu tổng hợp mới này? Với các biến dạng số, chúng ta sẽ thực hiện chính xác như những gì chúng ta làm với SMOTE. Còn các giá trị hạng mục được tính là các giá trị được hiển thị bởi phần lớn các neighbour.

- **Video:** [SMOTE-NC](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/23043550#overview)

ADASYN sử dụng phân phối có trọng số của lớp thiểu số để tạo dữ liệu tổng hợp mới, và phân phối này được tính trọng số tùy theo mức độ khó học hoặc khó phân loại của các quan sát. Vậy sẽ có nhiều dữ liệu tổng hợp hơn được tạo ra từ các mẫu khó phân loại hơn.

SMOTE sử dụng tất cả các mẫu của lớp thiểu số để tạo dữ liệu tổng hợp, ADASYN cũng sử dụng tất cả các mẫu của lớp thiểu số, nhưng nó sẽ sử dụng nhiều mẫu làm template của lớp thiểu số khó phân loại hơn và ít mẫu dễ phân loại hơn. Vì vậy, trong trường hợp SMOTE, các mẫu của lớp thiểu số để tạo dữ liệu tổng hợp sẽ không được chọn ngẫu nhiên. 

- **Video:** [ADASYN](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/23043554#overview)

SMOTE đường biên là một biến thể của SMOTE, chỉ tạo ra các ví dụ tổng hợp từ các quan sát trong lớp thiểu số gần hơn với ranh giới với (các) lớp đa số. SMOTE đường biên có hai biến thể, ở phiên bản 1, nó sẽ chỉ tạo neighbour từ lớp thiểu số, nhưng ở phiên bản 2, nó sẽ nội suy cả từ neighbour của lớp thiểu số và lớp đa số.

- **Video:** [SMOTE đường biên](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/23043562#overview)

Ở bước đầu tiên, SVM SMOTE sẽ huấn luyện một SVM trên toàn tập tập dữ liệu. Nó sẽ tìm các vectơ hỗ trợ. Chúng ta có các vectơ hỗ trợ của lớp thiểu số và lớp đa số, trong trường hợp này nó sẽ chỉ chọn các vectơ hỗ trợ từ lớp thiểu số. Và đây sẽ là khuôn mẫu cho dữ liệu tổng hợp.

Có hai phương pháp tạo dữ liệu tổng hợp. Chúng ta có phép nội suy như SMOTE và phép ngoại suy là điểm mới trong phương pháp này. Nếu vectơ hỗ trợ được bao quanh chủ yếu bởi các neighbour từ lớp thiểu số, nó sẽ tạo dữ liệu bằng phép ngoại suy (chúng ta sẽ mở rộng ranh giới của lớp thiểu số với phép ngoại suy). Nhưng nếu vectơ hỗ trợ được bao quanh chủ yếu bởi các neighbour từ lớp đa số, nó sẽ tạo dữ liệu bằng phép nội suy.

- **Video:** [SVM SMOTE](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/23043568#overview)

Ý tưởng của K-Means SMOTE là thúc đẩy các khu vực lớp thiểu số bằng cách tạo các mẫu trong các cụm xuất hiện tự nhiên của lớp thiểu số. Điều kiện tiên quyết để triển khai kỹ thuật này là cần có cụm tự nhiên của lớp thiểu số. Nếu tập dữ liệu không có các cụm này thì sử dụng kỹ thuật này không có tác dụng gì cả.

K-Means SMOTE sẽ xem xét các cụm trong lớp, chọn ra một vài cụm thỏa mãn các tiêu chí tỉ lệ nhất định và tránh tạo nhiễu bằng cách chỉ tạo mẫu trong các cụm đó. Trong thuật toán này, số cụm của K-Means, tỷ lệ mất cân bằng để lọc ra các cụm và số lượng neighbour mà chúng ta sử dụng để tạo mẫu là những siêu tham số cần được điều chỉnh và tối ưu.

- **Video:** [K-Means SMOTE](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/23043574#overview)

### 3. So sánh

Các mô hình khác nhau có xu hướng hoạt động tốt nhất trên các tập dữ liệu khác nhau. Vì vậy, khi thực hiện các dự án của mình, chúng ta cần thử nghiệm để biết kỹ thuật oversampling nào sẽ cho chất lượng mô hình tốt nhất. Chúng ta có rất nhiều công cụ được đề xuất, có thể sử dụng để nâng cao hiệu suất.

- **Video:** [So sánh các phương pháp oversampling](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/23043586#overview)

Khi chúng ta **khuếch đại những quan sát của lớp thiểu số** với oversampling, ngẫu nhiên hoặc bằng cách tạo mẫu mới, trong nhiều trường hợp, chúng ta cũng có thể khuếch đại nhiễu. Với các phương pháp Undersampling, chúng ta có thể **loại bỏ các quan sát nhiễu**, là những quan sát nằm ngoài ranh giới. Tuy nhiên, chúng ta cũng sẽ làm mất thông tin quan trọng khi undersample lớp đa số.

Liệu chúng ta có thể kết hợp những mặt tích cực của các phương pháp Oversampling và Undersampling để tạo tập dữ liệu tốt hơn không? Một số người cho rằng chúng ta có thể tận dụng ưu điểm của cả hai. Nếu chúng ta kết hợp kỹ thuật Oversampling và Undersampling cách thích hợp, chúng ta có thể sử dụng phương pháp **Oversampling để tạo nhiều quan sát hơn một cách ngẫu nhiên của lớp thiểu số (random generation - SMOTE)**, là lớp ít được biểu diễn trong tập dữ liệu. Sau đó, chúng ta có thể sử dụng phương pháp **Undersampling (NCR, ENN, Tomek Links, Instance Hardness)** **để** **loại bỏ những quan sát nhiễu trong tập thiểu số mới được tạo**, giúp hạn chế tác động của kỹ thuật Oversampling này đối với nhiễu, cũng như hạn chế tác động của kỹ thuật Undersampling với việc làm mất thông tin của lớp đa số.

- **Video:** [Kết hợp over và under-sampling](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/23171978#overview)

Trong video này, chúng ta sẽ so sánh việc kết hợp sử dụng các phương pháp Oversampling và Undersampling so với chỉ sử dụng kỹ thuật Oversampling và xem liệu quá trính này có làm tăng chất lượng của mô hình học máy hay không

- **Video:** [So sánh kết hợp over và under-sampling với oversampling](https://funix.udemy.com/course/machine-learning-with-imbalanced-data/learn/lecture/23171986#overview)



### Labs

- [Notebooks](https://drive.google.com/drive/folders/1SVdlaVeFIy5-nFEYY424ul9jhywyL1JM?usp=share_link)

  