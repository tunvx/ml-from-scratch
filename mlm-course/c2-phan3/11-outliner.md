# Xử lý ngoại lệ và co giãn đặc trưng



### 1. Xử lý ngoại lai

- **Video: [Các kỹ thuật xử lý ngoại lai](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16170445#overview)**

**1. Cắt tỉa**

Trimming/Truncation liên quan đến việc loại bỏ các outlier khỏi tập dữ liệu. Chúng ta chỉ cần quyết định một phép đo để xác định outlier. Đó có thể là phép xấp xỉ Gauss cho các biến được phân phối chuẩn hoặc quy tắc tiệm cận IQR cho các biến lệch.

- **Video: [Cắt tỉa ngoại lai](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16170449#overview)**

**2 Giới hạn**

Censoring (Kiểm duyệt) hoặc Capping (Giới hạn) là giới hạn max/min của phân phối tại một giá trị bất kỳ. Nói cách khác, những giá trị lớn hơn hoặc nhỏ hơn các giá trị được xác định tùy ý đều được kiểm duyệt. Capping có thể thực hiện ở cả 2 đầu hoặc 1 đầu phân phối còn tùy thuộc vào biến và người dùng.

**Ưu điểm:** không loại bỏ dữ liệu

**Hạn chế:** Có thể làm sai lệch các phân phối của biến và mỗi quan hệ giữa các biến.

- **Video: [Giới hạn ngoại lai với IQR](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8749894#overview)**

- **Video: [Giới hạn ngoại lai với trung bình và độ lệch chuẩn](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8749120#overview)**
- **Video: [Giới hạn ngoại lai với phân vị](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16170461#overview)**
- **Video: [Giới hạn tùy ý](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16170463#overview)**



### 2. Co giãn đặc trưng

Độ lớn của đặc trưng quan trọng vì:

- - Hệ số hồi quy của mô hình tuyến tính ảnh hưởng trực tiếp bởi tỉ lệ của các đặc trưng.
  - Các biến có độ lớn/phạm vi giá trị lớn hơn sẽ vượt trội hơn so với các biến có độ lớn/phạm vi giá trị nhỏ hơn.
  - Gradient descent hội tụ nhanh hơn khi các đặc trưng có cùng thang đo.
  - Co giãn đặc trưng giúp làm giảm thời gian tìm các vectơ hỗ trợ cho SVM.
  - Khoảng cách Euclid nhạy với độ lớn của đặc trưng.
  - Một số thuật toán như PAC yêu cầu các đặc trưng tập trung ở 0.

- **Video: [Co giãn đặc trưng](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16341806#overview)**



Chuẩn tắc hóa gồm căn giữa biến ở 0 và chuẩn tắc hóa phương sai thành 1. Quy trình là trừ đi mean của mỗi quan sát rồi chia cho độ lệch chuẩn: **z = (x - x_mean)/std**

Kết quả của phép biến đổi trên là z, được gọi là z-score thể hiện độ lệch chuẩn mà một quan sát nhất định lệch khỏi mean. z-score xác định vị trí của quan sát trong một phân phối (theo số lượng độ lệch chuẩn với giá trị trung bình của phân phối). Dấu của z-score (+ hoặc -) cho biết quan sát nằm trên (+) hay dưới (-) mean.

- **Video: [Chuẩn tắc hóa](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16366820#overview)**



Chuẩn hóa trung bình gồm việc căn giữa các biến ở 0 và điều chỉnh lại phạm vi giá trị. Quy trình gồm việc trừ đi giá trị trung bình của mỗi quan sát rồi chia cho hiệu giữa max và min: x_scaled = **(x - x_mean)/( x_max - x_min)**

Kết quả của phép biến đổi trên là một phân phối căn giữa 0 và min/max nằm trong phạm vi từ -1 đến 1. Hình dạng của phân phối chuẩn hóa trung bình sẽ tương tự như của phân phối ban đầu, nhưng phương sai có thể thay đổi nên sẽ không giống hệt nhau.

- **Video: [Chuẩn hóa trung bình](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16341810#overview)**



Co giãn về min-max các giá trị từ 0 đến 1. Nó trừ min từ tất cả các quan sát rồi chia cho phạm vi giá trị: X_scaled = (X - X.min / (X.max - X.min)

Kết quả của phép biến đổi trên là phân phối có các giá trị thay đổi trong phạm vi từ 0 đến 1. Nhưng giá trị trung bình không tập trung ở 0 và độ lệch chuẩn cũng thay đổi trong các biến. Hình dạng của phân phối khi co giãn max/min sẽ tương tự như phân phối ban đầu, nhưng phương sai có thể thay đổi nên chúng sẽ không giống nhau. Kỹ thuật co giãn này cũng nhạy với các outlier.

- **Video: [Co giãn về min-max](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16367234#overview)**



MaxAbsScaling co giãn dữ liệu thành giá trị tuyệt đối lớn nhất: **X_scaled = X / abs(X.max)**

Kết quả của phép biến đổi trên là phân phối có các giá trị thay đổi trong khoảng từ -1 đến 1, nhưng giá trị trung bình không căn giữa ở 0 và độ lệch chuẩn thay đổi trên các biến.

- **Video: [Co giãn về giá trị lớn nhất tuyệt đối](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16341818#overview)**



Trong quy trình này, median bị loại khỏi các quan sát rồi bị co lại lệ thành IQR. IQR là phạm vi giữa quartile thứ nhất (quantile thứ 25) và quartile thứ ba (quantile thứ 75): X_scaled = X - X_median / ( X.quantile(0.75) - X.quantile(0.25) )

Phương pháp RobustScaling này tạo ra các ước tính mạnh mẽ hơn cho trung tâm và phạm vi của biến, và được khuyến nghị dùng nếu dữ liệu hiển thị outlier.

- **Video: [Co giãn về trung vị và phân vị](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16367602#overview)**



Trong co giãn về độ dài vector đơn vị, chúng ta co giãn các thành phần của vectơ đặc trưng sao cho vectơ hoàn chỉnh có độ dài là 1, hoặc nói cách khác, có chuẩn là 1. Lưu ý rằng quy trình chuẩn hóa này sẽ chuẩn hóa vectơ đặc trưng chứ không phải vectơ quan sát. Vì vậy, chúng ta chia chuẩn của vectơ đặc trưng cho từng quan sát trên các biến khác nhau mà không phải chia chuẩn của vectơ quan sát cho các quan sát có cùng đối tượng.

Co giãn về vectơ đơn vị được tính bằng cách chia từng vectơ đặc trưng cho khoảng cách Manhattan (chuẩn l1) hoặc khoảng cách Euclid của vectơ (chuẩn l2):

- - X_scaled_l1 = X / l1(X)
  - X_scaled_l2 = X / l2(X)

Khoảng cách Manhattan là tổng các thành phần tuyệt đối của vectơ: **l1(X) = |x1| + |x2| + ... + |xn|**

Còn khoảng cách Euclid được tính bằng căn bậc hai của tổng các thành phần của vectơ: **l2(X) = sqr( x1^2 + x2^2 + ... + xn^2 )**

- **Video: [Co giãn về độ dài vector đơn vị](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16341822#overview)**



### 3. Các loại biến khác

Chúng ta sẽ thảo luận về cách thiết kế biến hỗn hợp, là biến chứa cả số và string trong giá trị.

- ***\*Video:\** [Biến hỗn hợp](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8903194#overview)**

Biến thời gian là những biến có chứa ngày, giờ hoặc tổ hợp của cả hai mà chúng ta gọi là timestamp (dấu thời gian) hoặc date-timestamp. Nhìn chung, chúng ta sẽ không sử dụng chúng như vậy khi xây dựng các mô hình học máy mà cần trích xuất một loạt các đặc trưng từ các đặc điểm này.

- ***\*Video:\** [Biến thời gian](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16421654#overview)**

Chúng ta sẽ xem xét cách trích xuất giờ, phút, giây và thời gian trôi qua giữa hai biến với các múi giờ khác nhau.

- ***\*Video:\** [Biến thời gian và múi giờ khác nhác nhau](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16386540#overview)**

### Lab 10: Co dãn đặc trưng

- [Dataset](https://drive.google.com/file/d/17QvEIOCNPUz3WHdvf3vwbHoDj_O5w8Hd/view)
- [Notebooks](labs/lab10.zip)