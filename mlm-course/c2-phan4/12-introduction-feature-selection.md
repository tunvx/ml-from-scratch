# Giới thiệu về Lựa chọn đặc trưng

- [Lựa chọn đặc trưng là gì?](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/9341700#overview)



## 1. Tổng quan về lựa chọn đặc trưng

- **Video: [Các phương pháp lựa chọn đặc trưng](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/9341702#overview)**

Phương pháp lọc là các thủ tục lựa chọn đặc trưng dựa trên các đặc điểm của dữ liệu. Đây là đặc điểm của chính các đặc trưng để lựa chọn các biến. Phương pháp lọc không hề liên quan đến các thuật toán học máy tại thời điểm sàng lọc các đặc trưng. Phương pháp này chỉ đánh giá các đặc trưng và đưa ra lựa chọn dựa trên các đặc điểm của đặc trưng. 

Vì những lý do này nên phương pháp lọc ít tốn kém về mặt tính toán hơn bất kỳ thủ tục lựa chọn đặc trưng khác. Tuy nhiên, chúng thường cho dự đoán có chất lượng thấp hơn so với phương pháp gói hay phương pháp nhúng. Mặt khác, phương pháp lọc rất phù hợp với quick screen và nhanh chóng loại bỏ các đặc trưng không liên quan khỏi tập dữ liệu.

- **Video:** [Phương pháp lọc](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/9341704#overview)

Thay vào đó, phương pháp gói sử dụng thuật toán học máy dự đoán để lựa chọn tập hợp con đặc trưng tối ưu. Về bản chất, phương pháp gói sẽ xây dựng một thuật toán học máy cho từng tập hợp con đặc trưng mà chúng đánh giá, sau đó chọn tập hợp con của các biến tạo ra thuật toán có chất lượng cao nhất. Tuy nhiên, chúng cung cấp tập hợp con các đặc trưng hoạt động tốt nhất cho thuật toán học máy cụ thể mà chúng sử dụng để lựa chọn không gian đặc trưng.

- **Video:** [Phương pháp gói](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/9341706#overview)

Phương pháp nhúng thực hiện lựa chọn đặc trưng như một phần của quá trình xây dựng mô hình học máy. Bằng cách kết hợp lựa chọn đặc trưng với xây dựng hồi quy hoặc phân loại, phương pháp nhúng có ưu điểm của mô hình gói ở chỗ chúng xem xét tương tác giữa mô hình học máy và các đặc trưng. Phương pháp nhúng cũng ít cần tính toán chuyên sâu hơn phương pháp gói vì chúng không khớp các mô hình khác nhau với các tập hợp con các đặc trưng khác nhau, thay vào đó chúng chỉ xây dựng một mô hình học máy duy nhất và lưạ chọn các đặc trưng dựa trên mức độ quan trọng của chúng với thuật toán. 

- **Video:** [Phương pháp nhúng](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/9341708#overview)

Trong phần 4 này, chúng ta sẽ tìm hiểu về các phương pháp lựa chọn đặc trưng dựa trên Pandas, Numpy để tự xây dựng mô hình lựa chọn hoặc sử dụng các thư viện mã nguồn mở như scikit-learn và Feature-engine.

- **Video:** [Thư viện mã nguồn mở cho lựa chọn đặc trưng](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/24061034#overview)



## 2. Phương pháp gói

- **Video**: [Chi tiết về phương pháp gói](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/9341766#overview)

Lựa chọn đặc trưng theo các phương pháp xuôi bắt đầu bằng cách huấn luyện mô hình học máy cho từng đặc trưng trong tập dữ liệu và lựa chọn đặc trưng mở đầu khiến mô hình hoạt động tốt nhất theo tiêu chí đánh giá nhất định.

Ở bước thứ hai, nó tạo ra các mô hình học máy cho tất cả các tổ hợp đặc trưng đã chọn ở bước trước và đặc trưng thứ hai. Nó chọn cặp tạo ra thuật toán hoạt động tốt nhất.

Phương pháp này tiếp tục bằng cách thêm mỗi lần 1 đặc trưng vào các đặc trưng đã chọn ở các bước trước cho đến khi xác định trước tiêu chí dừng.

- **Video**: [Lựa chọn tính năng theo các phương pháp xuôi](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22544880#overview)

Lựa chọn đặc trưng theo các phương pháp ngược bắt đầu bằng cách khớp mô hình học máy sử dụng tất cả các đặc trưng trong tập dữ liệu và xác định chất lượng mô hình.

Sau đó, nó huấn luyện mô hình trên tất cả các tổ hợp có thể có của tất cả các đặc trưng - 1, loại bỏ đặc trưng trả về mô hình có chất lượng cao nhất khi bỏ đặc trưng đó đi.

Ở bước thứ ba, huấn luyện các mô hình trong tất cả các tổ hợp có thể của các đặc trưng còn lại từ bước hai bớt đi 1 đặc trưng và loại bỏ đặc trưng khiến mô hình hoạt động tốt nhất.

Thuật toán dừng theo một tiêu chí do người dùng xác định. Tiêu chí này có thể là chất lượng mô hình không giảm vượt quá một ngưỡng nhất định hoặc đạt tới số lượng đặc trưng đã chọn nhất định như trong triển khai mlxtend.

- **Video**: [Lựa chọn tính năng theo các phương pháp ngược](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22544882#overview)

Tìm kiếm đầy đủ tìm tập hợp con các đặc trưng tốt nhất trong số tất cả các tập hợp con đặc trưng có thể theo một phép đo đặc trưng xác định cho một thuật toán học máy nhất định. 

Tìm kiếm đầy đủ đánh giá tất cả các kết hợp đặc trưng có thể có. Nó rất khó tính toán và thậm chí là không khả thi nếu không gian đặc trưng lớn.

- **Video**: [Tìm kiểm đầy đủ](https://funix.udemy.com/course/feature-selection-for-machine-learning/learn/lecture/22544886#overview)