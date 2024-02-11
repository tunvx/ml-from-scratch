# Các đặc điểm của biến

- **Video: [Các đặc điểm của biến](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/10199080#overview)**



## 1. Dữ liệu bị khuyết

Có 3 cơ chế chính dẫn đến việc dữ liệu bị khuyết:

- - **MCAR**: Dữ liệu bị khuyết một cách ngẫu nhiên, với xác suất bị thiếu là như nhau ở các quan sát.
  - **MAR**: Có mối quan hệ hệ thống giữa các khuynh hướng của các giá trị bị khuyết và dữ liệu quan sát được. Nói cách khác, xác suất một quan sát bị khuyết phụ thuộc vào bệnh thông tin sẵn có, nó độc lập với các biến khác trong tập dữ liệu.
  - **MNAR**: Việc thiếu các giá trị không phải là ngẫu nhiên nếu có một cơ chế hoặc một lý do nào đó khiến các giá trị bị khuyết được đưa vào tập dữ liệu.

- Video: [Dữ liệu bị khuyết](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/7690396#overview)



## 2. Các biến hạng mục

- Video:[Cardinality - các biến hạng mục](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/7690414#overview)
- Video: [Nhãn hiếm - các biến hạng mục](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/7690400#overview)



## 3. Các giả định về quan hệ tuyến tính

2 biến có quan hệ tuyến tính nếu thỏa mãn các giả định sau:

- - Độ tuyến tính (linearity): Phương trình hồi quy sẽ là (hoặc có dạng gần đúng là) đường thẳng.
  - Không có mối quan hệ tuyến tính hoàn hảo giữa các biến: Giúp chúng ta phát hiện ra có xuất hiện mối quan hệ ảo giữa các biến, hay biến này được tạo ra từ biến kia.
  - Sai số có phân phối chuẩn.
  - Phương sai của các phần dư không đổi.

Bên cạnh việc dựng lên các đường hồi quy tuyến tính và kiểm định các giả định, chúng ta cũng có thể quan sát ma trận tương quan để quan sát về mối quan hệ tương quan tuyến tính giữa các biến.

- Video: [Các giả định về quan hệ tuyến tính](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/7690428#overview)



## 4. Phân phối biến và biến đổi biến

**1. Phân phối biến**

Trong video này, chúng ta sẽ lại một vài các phân phối biến hay được sử dụng trong xác suất thống kê, đặc biệt là phân phối chuẩn.

Việc các biến có tuân theo phân phối chuẩn hay phân phối lệch có vai trò quan trọng trong việc xác định các phương pháp gán giá trị bị khuyết cho biến. Nếu biến được phân phối chuẩn, chúng ta thường chọn bù các giá trị bị khuyết với giá trị mean, còn nếu biến bị lệch rất có thể chúng ta sẽ bù các giá trị bị khuyết với median vì median cho các phân phối lệch đại diện cho số đông.

Có thể sử dụng các biến đổi toán học khác nhau đưa biến có phân phối lệch về phân phối chuẩn, chúng ta sẽ đi sâu vào vấn đề này hơn ở video dưới.

- Video: [Phân phối biến](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/7690416#overview)

**2. Biến đổi biến**

Các phép biến đổi toán học hay được sử dụng cho biến đổi biến:

- - Biến đổi logarit.
  - Biến đổi nghịch đảo
  - Biến đổi hàm mũ hay lũy thừa
  - Biến đổi hàm mũ đặc biệt: Box-Cox (chỉ xác định với giá trị dương), Yeo-Johnson (với hệ số lũy thừa λ).

- Video: [Biến đổi biến](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/16120743#overview)



## 5. Ngoại lai

Với phân phối lệch, outlier sẽ bé hơn Q1 - 1.5 * IQR hoặc lớn hơn Q3 + 1.5 * IQR. Ở phía trải rộng hơn của phân phối, chúng ta có thể thay thế 1.5 bằng 3 hoặc một con số phù hợp khác.

- [Ngoại lai](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/7690406#overview)



## 6. Độ lớn biến

- Video:[Độ lớn biến](https://funix.udemy.com/course/feature-engineering-for-machine-learning/learn/lecture/7690422#overview)

## 7. Labs: Làm quen với xử lý đặc trưng

- [Dataset](https://drive.google.com/file/d/17QvEIOCNPUz3WHdvf3vwbHoDj_O5w8Hd/view)
- [Notebooks](labs/lab7.zip)