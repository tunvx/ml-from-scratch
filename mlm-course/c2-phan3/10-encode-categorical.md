# Mã hoá biến hạng mục

- Video: [Mã hóa biến hạng mục](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16120489#overview)



## 1. Mã hóa one-hot

- ***\*Video:\** [Mã hóa one-hot](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8885342#overview)**
- ***\*Video:\** [Mã hóa one-hot cho các hạng mục hàng đầu](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8885522#overview)**



## 2. Mã hóa theo xác suất - thống kê

- ***\*Video:\** [Mã hóa thứ tự/mã hóa nhãn](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8885652#overview)**
- ***\*Video:\** [Mã hóa đếm/tần số](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8886224#overview)**



## 3. Mã hóa mục tiêu

- ***\*Video:\** [Mã hóa thứ tự có hướng dẫn mục tiêu](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8886574#overview)**
- ***\*Video:\** [Mã hóa trung bình](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8886682#overview)**



## 4. Mã hóa nâng cao (tham khảo)

Trọng số bằng chứng (WoE) được phát triển chủ yếu cho ngành tài chính và tín dụng, giúp xây dựng nhiều mô hình có tính dự báo hơn để đánh giá rủi ro vỡ nợ. Nó dự đoán khả năng số tiền cho một người hoặc tổ chức vay bị mất. Như vậy, trọng số bằng chứng là thước đo “sức mạnh” của kỹ thuật phân nhóm để phân tách rủi ro tốt và xấu (vỡ nợ).

- - WoE sẽ bằng 0 nếu P (Goods)/P (Bads) = 1, nghĩa là, nếu kết quả là ngẫu nhiên cho nhóm đó.
  - Nếu P(Bads) > P(Goods) thì tỷ lệ chênh lệch sẽ < 1 
  - WoE sẽ < 0 nếu P (Goods)> P (Bads).

- ***\*Video:\** [WoE - trọng số bằng chứng](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8890280#overview)**



Giá trị hiếm là các hạng mục trong một biến hạng mục chỉ xuất hiện trong tỷ lệ nhỏ các quan sát. Không có quy tắc chung nào để xác định thế nào là tỷ lệ phần trăm nhỏ. Thông thường, bất kỳ giá trị nào dưới 5% đều có thể coi là hiếm.

Như chúng ta đã thảo luận trong bài học trước, các nhãn không thường xuất hiện rất ít, do đó rất khó để lấy được thông tin đáng tin cậy từ chúng. Nhưng quan trọng hơn, các nhãn không thường xuất hiện có xu hướng chỉ xuất hiện trên tập huấn luyện hoặc chỉ trên tập kiểm tra:

Nếu chỉ trên tập huấn luyện, chúng có thể gây ra overfitting. Nếu chỉ trên tập kiểm tra, mô hình học máy sẽ không biết cách cho tính chúng

Do đó, để tránh cách xử lý này, chúng ta có xu hướng nhóm chúng vào một hạng mục mới là 'Rare' hoặc 'Other'.

- ***\*Video:\** [Mã hóa nhãn hiếm](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/8793662#overview)**



Mã hóa nhị phân sử dụng code nhị phân là tổ hợp của các số 0 và 1 để mã hóa ý nghĩa của biến. 

Hàm băm đặc trưng là một phương pháp thay thế cho phép giảm kích thước trong khi mã hóa các giá trị hạng mục và nó hoạt động như sau:

- - Đầu tiên, chúng ta cần quyết định tùy ý xem sẽ lấy bao nhiêu biến từ mỗi hạng mục.
  - Sau đó, chúng ta cần tạo phương thức hash.
  - Với hàm băm bạn tạo, hãy lấy các số từ nhãn.

- ***\*Video:\** [Mã hóa nhị phân và áp dụng hàm băm](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16120631#overview)**



### Lab 9: Mã hóa biến hạng mục cơ bản

- [Dataset](https://drive.google.com/file/d/17QvEIOCNPUz3WHdvf3vwbHoDj_O5w8Hd/view)
- [Notebooks](labs/lab9.zip)
